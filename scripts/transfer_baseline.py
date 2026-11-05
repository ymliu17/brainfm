#!/usr/bin/env python3
"""Put the trivial baselines through the SAME ADNI -> RCT chain the foundation model uses.

WHY
---
Frozen BrainLM already loses to a 424-dim vector of per-ROI standard deviations on every
ADNI covariate, and to FC edges on most. If the FM is then given an ADNI fine-tuning stage
and the baselines are not, an FM win on the RCTs says nothing: it could be the ADNI data,
not the representation. So the baselines get the same three-step chain:

    1. compute a feature on the ADNI paired build   (FC edges / amplitude profile)
    2. fit a SUPERVISED PROJECTION on ADNI against the ADNI outcome
    3. apply that projection to the RCT features, then cross-validate a head on the RCT

Step 2 is the analogue of fine-tuning: an ADNI-learned map from features to a low-dim
embedding, which the RCT head then reads. PLS regression is used because it is the
supervised dimensionality reduction whose `transform` is a fixed linear map, so "the
representation ADNI learned" is unambiguous and transfers without refitting.

`--source none` is the control that makes the comparison interpretable: skip ADNI entirely
and cross-validate the head on raw RCT features. If `--source none` matches or beats the
ADNI-projected version, the ADNI stage contributed nothing for that feature -- and the same
question must then be asked of the FM.

WHY THESE FEATURES TRANSFER AT ALL
----------------------------------
FC (424x424 correlations) and the amplitude profile (424 per-ROI sds) have a dimension set
by the atlas, not by the scan length. So ADNI at T=140 and ACT at T=300 and CogTE at T=118
all produce the same feature space with no resampling, no cropping and no padding. The
foundation model cannot do this: it needs a fixed token count, which is what forces the
temporal-geometry choices that `the original cohort audit` records as a confound.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.fc_features import compute_fc, upper_tri  # noqa: E402
from cohorts import load_cohort  # noqa: E402
from paths import COHORT, DERIVED_DIR, RUNS_DIR  # noqa: E402

#: How the two scans' features are combined into one vector per subject. Mirrors
#: adni_finetune.BrainLMPairedRegressor.COMBINE so the FM and the baselines are asked the
#: same question. {diff, mean} is an orthogonal reparameterisation of {T1, T2}.
COMBINES = ("diff", "mean", "concat", "t1", "t2")


def combine(f1: np.ndarray, f2: np.ndarray, how: str) -> np.ndarray:
    if how == "diff":
        return f2 - f1
    if how == "mean":
        return 0.5 * (f1 + f2)
    if how == "t1":
        return f1
    if how == "t2":
        return f2
    return np.hstack([f1, f2])


def fc_feature(x: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Fisher-z'd upper-triangle FC. [N, R, T] -> [N, R(R-1)/2]."""
    fc = compute_fc(x, valid)
    e = upper_tri(fc)
    # Fisher z: correlations near +-1 are compressed on the r scale, and the head is linear.
    return np.arctanh(np.clip(e, -0.999999, 0.999999))


def amp_feature(x: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """log per-ROI sd divided by the scan's median sd. [N, R, T] -> [N, R].

    Dividing by the scan's own median makes it invariant to the arbitrary BOLD gain, so it
    carries only the BETWEEN-ROI amplitude structure -- exactly the part per-ROI z-scoring
    destroys, and the part that out-predicts frozen BrainLM on ADNI covariates.
    """
    out = np.empty((len(x), x.shape[1]))
    for i in range(len(x)):
        run = x[i, :, : int(valid[i])] if valid is not None else x[i]
        sd = run.std(axis=1)
        out[i] = np.log(np.maximum(sd / max(np.median(sd), 1e-12), 1e-6))
    return out


FEATURES = {"fc": fc_feature, "amp": amp_feature}


def build(x4: np.ndarray, valid: np.ndarray | None, feature: str, how: str) -> np.ndarray:
    fn = FEATURES[feature]
    v1 = valid[:, 0] if valid is not None else None
    f1 = fn(x4[..., 0], v1)
    if how == "t1":
        return f1
    v2 = valid[:, 1] if valid is not None else None
    f2 = fn(x4[..., 1], v2)
    return combine(f1, f2, how)


def load_adni(tag: str, target: str, in_dir: Path):
    x = torch.load(in_dir / f"ADNI_x_{tag}.pt", map_location="cpu", weights_only=False)
    y = torch.load(in_dir / f"ADNI_d{target}_{tag}.pt", map_location="cpu",
                   weights_only=False).numpy()
    man = pd.read_csv(in_dir / f"ADNI_manifest_{tag}.csv")
    ok = np.isfinite(y)
    return x[torch.as_tensor(ok)].numpy(), y[ok], man.loc[ok].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="ADNI->RCT transfer for the trivial baselines")
    ap.add_argument("--cohort", required=True, choices=sorted(COHORT))
    ap.add_argument("--label", default="mem", choices=("mem", "ef"))
    ap.add_argument("--outcome", default="binary", choices=("binary", "continuous"),
                    help="binary: the improvement indicator, scored by AUC. continuous: the "
                         "change score, scored by Spearman rho -- keeps the magnitude that "
                         "binarising throws away, which matters at n<100.")
    ap.add_argument("--feature", default="fc", choices=sorted(FEATURES) + ["fc+amp"])
    ap.add_argument("--combine", default="diff", choices=COMBINES)
    ap.add_argument("--source", default="paired", choices=("paired", "none"),
                    help="paired: fit the PLS projection on the ADNI (T1,T2)->change task. "
                         "none: no ADNI stage at all, head fitted on raw RCT features.")
    ap.add_argument("--adni_tag", default="raw_first_last_t140_off")
    ap.add_argument("--adni_target", default=None,
                    help="ADNI outcome to fit the projection against; defaults to --label")
    ap.add_argument("--adni_dir", default=str(DERIVED_DIR / "adni_paired"))
    ap.add_argument("--n_components", type=int, default=16)
    ap.add_argument("--rct_release", default="amp", choices=("amp", "zscore"),
                    help="which RCT imaging FILE to read; not a normalisation choice")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default=str(RUNS_DIR / "v2" / "transfer_baseline"))
    args = ap.parse_args()

    co = load_cohort(args.cohort, release=args.rct_release)
    mask = co.labelled_mask(args.label)
    xr = co.bold[torch.as_tensor(mask)].numpy()
    vr = co.valid[torch.as_tensor(mask)].numpy()
    y_bin = co.labels(args.label)[mask]
    y_cont = co.change_scores(args.label)[mask]
    y = y_bin if args.outcome == "binary" else y_cont
    fold_col = COHORT[args.cohort]["fold_col_for_label"][args.label]
    folds = co.folds(fold_col)[mask].astype(int)
    kind = "bin" if args.outcome == "binary" else "cont"
    print(f"{args.cohort}/{args.label}/{args.outcome}: n={int(mask.sum())} "
          f"({'base rate %.3f' % y_bin.mean() if kind == 'bin' else 'sd %.3f' % np.nanstd(y_cont)})"
          f"  folds {sorted(set(folds))}  rct_release={args.rct_release}", flush=True)

    feats = ["fc", "amp"] if args.feature == "fc+amp" else [args.feature]
    Fr = np.hstack([build(xr, vr, f, args.combine) for f in feats])
    print(f"  RCT feature {args.feature}/{args.combine}: {Fr.shape}", flush=True)

    proj = None
    if args.source == "paired":
        tgt = args.adni_target or args.label
        xa, ya, mana = load_adni(args.adni_tag, tgt, Path(args.adni_dir))
        Fa = np.hstack([build(xa, None, f, args.combine) for f in feats])
        if Fa.shape[1] != Fr.shape[1]:
            raise SystemExit(f"feature width differs: ADNI {Fa.shape[1]} vs RCT {Fr.shape[1]}")
        print(f"  ADNI {args.adni_tag} d{tgt}: {Fa.shape} over "
              f"{mana['PTID'].nunique()} subjects", flush=True)
        # The projection is fitted on ADNI ONLY. It never sees an RCT label, so applying it
        # to every RCT row -- including held-out ones -- leaks nothing.
        sa = StandardScaler().fit(Fa)
        k = min(args.n_components, Fa.shape[0] - 1, Fa.shape[1])

        # Honest ADNI-side score, subject-grouped. This is the number to compare against
        # the FM's stage-1 cross-validated rho: it says whether the ADNI stage learned
        # anything at all. The in-sample rho printed next to it is reported only to show
        # HOW MUCH the projection overfits -- at k=16 on a few hundred subjects it can
        # approach 1.0 while the grouped-CV value sits at zero, and in that case the
        # projection is a set of directions fitted to ADNI noise. The RCT transfer below is
        # still valid (no RCT label is involved), it just has nothing to transfer.
        subs = mana["PTID"].values
        uniq = np.unique(subs)
        rs = np.random.default_rng(0).permutation(len(uniq))
        gf = {su: (i % 5) + 1 for i, su in enumerate(uniq[rs])}
        gfold = np.array([gf[su] for su in subs])
        oof = np.full(len(ya), np.nan)
        for kk in range(1, 6):
            trm, tem = gfold != kk, gfold == kk
            if trm.sum() < k + 2:
                continue
            s_in = StandardScaler().fit(Fa[trm])
            pl = PLSRegression(n_components=min(k, int(trm.sum()) - 1),
                              scale=False).fit(s_in.transform(Fa[trm]), ya[trm])
            oof[tem] = pl.predict(s_in.transform(Fa[tem])).ravel()
        vv = np.isfinite(oof)
        adni_cv = float(stats.spearmanr(ya[vv], oof[vv]).statistic) if vv.sum() > 10 else float("nan")

        pls = PLSRegression(n_components=k, scale=False).fit(sa.transform(Fa), ya)
        proj = (sa, pls)
        r = stats.spearmanr(ya, pls.predict(sa.transform(Fa)).ravel()).statistic
        print(f"  PLS k={k} on ADNI: grouped-CV rho {adni_cv:+.4f}  "
              f"(in-sample {r:+.4f} -- the gap is the overfit)", flush=True)
        Fr = pls.transform(sa.transform(Fr))
        print(f"  RCT after ADNI projection: {Fr.shape}", flush=True)

    grid = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    scores = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        pred = np.full(len(y), np.nan)
        for k in sorted(set(folds)):
            te = folds == k
            tr = ~te
            if tr.sum() < 20 or te.sum() < 3:
                continue
            # Hyperparameter on an inner split of the training folds only.
            idx = np.where(tr)[0]
            perm = rng.permutation(len(idx))
            cut = max(1, int(0.2 * len(idx)))
            iv, it = idx[perm[:cut]], idx[perm[cut:]]
            sc = StandardScaler().fit(Fr[it])
            best, best_a = -np.inf, grid[0]
            for a in grid:
                if kind == "bin":
                    if len(set(y[it])) < 2 or len(set(y[iv])) < 2:
                        continue
                    m = LogisticRegression(C=a, max_iter=5000).fit(sc.transform(Fr[it]), y[it])
                    s = roc_auc_score(y[iv], m.predict_proba(sc.transform(Fr[iv]))[:, 1])
                else:
                    m = Ridge(alpha=1.0 / a).fit(sc.transform(Fr[it]), y[it])
                    s = stats.spearmanr(y[iv], m.predict(sc.transform(Fr[iv]))).statistic
                    s = s if np.isfinite(s) else -np.inf
                if s > best:
                    best, best_a = s, a
            sc = StandardScaler().fit(Fr[tr])
            if kind == "bin":
                m = LogisticRegression(C=best_a, max_iter=5000).fit(sc.transform(Fr[tr]), y[tr])
                pred[te] = m.predict_proba(sc.transform(Fr[te]))[:, 1]
            else:
                m = Ridge(alpha=1.0 / best_a).fit(sc.transform(Fr[tr]), y[tr])
                pred[te] = m.predict(sc.transform(Fr[te]))
        v = np.isfinite(pred) & np.isfinite(y)
        scores.append(roc_auc_score(y[v], pred[v]) if kind == "bin"
                      else float(stats.spearmanr(y[v], pred[v]).statistic))

    metric = "auc" if kind == "bin" else "spearman"
    ref = max(y_bin.mean(), 1 - y_bin.mean()) if kind == "bin" else 0.0
    print(f"\n  {metric} = {np.mean(scores):+.4f} +/- {np.std(scores):.4f}   "
          f"seeds {np.round(scores, 4).tolist()}")
    print(f"  reference: {'majority-class accuracy %.3f (AUC 0.5)' % ref if kind == 'bin' else 'rho 0'}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = (f"{args.cohort}_{args.label}_{args.outcome}_{args.feature}_{args.combine}"
            f"_{args.source}_rct-{args.rct_release}")
    json.dump({"args": vars(args), metric: [float(np.mean(scores)), float(np.std(scores))],
               "per_seed": [float(s) for s in scores], "n": int(mask.sum()),
               "adni_grouped_cv_rho": (adni_cv if args.source == "paired" else None)},
              open(out / f"{stem}.json", "w"), indent=1)
    print(f"  wrote {out / (stem + '.json')}")


if __name__ == "__main__":
    main()
