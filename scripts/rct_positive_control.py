#!/usr/bin/env python3
"""Can ANYTHING be decoded from the RCT imaging? The missing control.

WHY THIS HAD TO EXIST
---------------------
66 cells of RCT-side probing came back null: baseline cognitive level, continuous change,
and binary responder, across BrainLM, FC, the amplitude profile and co-skewness FC, in three
cohorts. That result has two readings and no earlier run could separate them:

    (a) these cohorts carry no decodable COGNITION signal at n = 74-107, or
    (b) these imaging tensors carry no decodable signal AT ALL -- a parcellation,
        alignment or subject-ordering defect would produce exactly the same 66 nulls.

scripts/positive_control.py only ever ran on ADNI (--x_adni / --manifest), so (b) was never
excluded. Age, sex and acquisition site are the standard controls: all three are decodable
from resting-state FC in intact data, none of them is the study's outcome, and site in
particular should be near-ceiling because it is an acquisition property rather than a
biological one. If site cannot be decoded from ACT's three scanners, the 66 nulls say
nothing about cognition.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import config as C  # noqa: E402
from cohorts import load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, RCT_DIR, RUNS_DIR, XYZ_PATH  # noqa: E402
from probe_readouts import brainlm_readout_batched  # noqa: E402
from transfer_baseline import amp_feature, fc_feature  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def act_controls(ids):
    t = pd.read_csv(DERIVED_DIR / "rct" / "act_covariates.csv").set_index("ID")
    t = t.reindex(ids)
    return {"age": (pd.to_numeric(t["formj_subject_age"], errors="coerce").values, "cont"),
            "sex": (pd.to_numeric(t["formj_subj_gender"], errors="coerce").values - 1, "bin"),
            "site": (pd.Categorical(t["study_site"]).codes.astype(float), "multi")}


def breathe_controls(ids):
    cog = pd.read_csv(RCT_DIR / "breathe" / "breathe_cognition0323.csv")
    cog = cog[cog["session_num"] == 1].copy()
    cog["ID"] = "sub-" + cog["subject_id"].astype(str).str.lstrip("s")
    t = cog.drop_duplicates("ID").set_index("ID").reindex(ids)
    sex = pd.Categorical(t["sex"]).codes.astype(float)
    return {"age": (pd.to_numeric(t["age"], errors="coerce").values, "cont"),
            "sex": (np.where(np.asarray(t["sex"].isna()), np.nan, sex), "bin")}


def cogte_controls(ids):
    """Age is DM2 and sex is DM1 in CogTE's analysis table (n=77, age 60-90, sex coded 1/2).

    Cortical thickness is carried too: it is a STRUCTURAL measure from the same
    participant's MRI, so it is a second independent control alongside age and sex. Baseline
    MOCA is deliberately NOT used -- it shares construct variance with the delayed-recall
    outcome, so it would be a same-family reference rather than a control.
    """
    t = pd.read_csv(RCT_DIR / "cogte" / "ITT_CogTE_analysis data_unstructure.csv")
    t["ID"] = "sub-" + t["ID"].astype(int).astype(str)
    t = t.drop_duplicates("ID").set_index("ID").reindex(ids)
    return {"age": (pd.to_numeric(t["DM2"], errors="coerce").values, "cont"),
            "sex": (pd.to_numeric(t["DM1"], errors="coerce").values - 1, "bin"),
            "cortical_thickness": (pd.to_numeric(t["average4CT_Bsl"],
                                                 errors="coerce").values, "cont")}


def psopt_controls(ids):
    """psopt's controls come from its own `labels.csv`, which prepare_psopt.py built.

    Cortical thickness is carried, as for CogTE, and here it is the AD-signature measure
    (`adsct_jack`, the Jack et al. temporal composite) rather than a whole-brain average --
    a structural measure from the same participant's MRI, so an independent control
    alongside age and sex.

    `mean_fd` is the control this cohort adds that no other one has: head motion is the
    nuisance most likely to be decodable from FC, so a cohort where it can be measured is
    the place to check whether an imaging-side effect is motion. It is taken at T1 to match
    the T1 features every arm uses.

    MoCA / GDS-30 / STAI / AES-18 are deliberately NOT controls. MoCA shares construct
    variance with the delayed-recall outcome (the reason it is excluded for CogTE too), and
    the affective scales are candidate OUTCOMES for this trial, not nuisances.
    """
    t = pd.read_csv(RCT_DIR / "psopt" / "labels.csv").set_index("ID").reindex(ids)
    return {"age": (pd.to_numeric(t["age"], errors="coerce").values, "cont"),
            "sex": (pd.to_numeric(t["sex_code"], errors="coerce").values, "bin"),
            "cortical_thickness": (pd.to_numeric(t["adsct_jack_bsl"],
                                                 errors="coerce").values, "cont"),
            "mean_fd": (pd.to_numeric(t["mean_fd_T1"], errors="coerce").values, "cont")}


CONTROLS = {"act": act_controls, "breathe": breathe_controls, "cogte": cogte_controls,
            "psopt": psopt_controls}


#: Same grid rct_probe.py selects over. A FIXED strength is not acceptable here: with
#: C=0.01 and no inner selection this script reported ACT sex at AUC 0.474 and ACT site at
#: exactly 0.3333 with zero variance across seeds -- the classifier collapsing to the
#: majority class, which reads as "the data carries nothing" when it is only the probe
#: giving up. Selecting C on the OUTER test folds is not a fix either (that gave sex 0.614
#: and was withdrawn); the strength must be chosen inside the training fold, as below.
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _eval(y, p, mode):
    if mode == "cont":
        r = stats.spearmanr(y, p).statistic
        return float(r) if np.isfinite(r) else float("nan")
    if mode == "bin":
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return float("nan")
    return float(balanced_accuracy_score(y, p))


def score(F, y, mode, seeds=(0, 1, 2)):
    """Nested: the regularisation strength is chosen on an inner split of each training
    fold, exactly as scripts/rct_probe.py does, so a null here is comparable to the 66
    nulls that motivated this script."""
    strat = (y.astype(int) if mode in ("bin", "multi")
             else np.clip(np.searchsorted(np.quantile(y, [.2, .4, .6, .8]), y), 0, 4))

    def fit_predict(Ftr, ytr, Fte, c, n_pca=100):
        # Standardise THEN reduce, fitted on the training rows only. The reverse order
        # (reduce globally, then StandardScaler the components) rescales every component
        # to unit variance, flattening the spectrum so that the trailing noise components
        # weigh as much as the leading signal ones. Measured on ACT sex with 94 components
        # from 95 samples: AUC 0.539 under the old order, 0.814 under this one.
        sc = StandardScaler().fit(Ftr)
        A, B = sc.transform(Ftr), sc.transform(Fte)
        if A.shape[1] > n_pca:
            pc = PCA(n_components=min(n_pca, len(A) - 1), random_state=0).fit(A)
            A, B = pc.transform(A), pc.transform(B)
        if mode == "cont":
            return Ridge(alpha=1.0 / c).fit(A, ytr).predict(B)
        m = LogisticRegression(C=c, max_iter=5000).fit(A, ytr)
        return (m.predict_proba(B)[:, 1] if mode == "bin" else m.predict(B))

    out = []
    for seed in seeds:
        oof = np.full(len(y), np.nan)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, strat):
            best, best_s = GRID[0], -np.inf
            for c in GRID:
                ip = np.full(len(tr), np.nan)
                for itr, ite in StratifiedKFold(3, shuffle=True,
                                                random_state=seed).split(F[tr], strat[tr]):
                    ip[ite] = fit_predict(F[tr][itr], y[tr][itr], F[tr][ite], c)
                v = _eval(y[tr], ip, mode)
                if np.isfinite(v) and v > best_s:
                    best, best_s = c, v
            oof[te] = fit_predict(F[tr], y[tr], F[te], best)
        out.append(_eval(y, oof, mode))
    return float(np.mean(out)), float(np.std(out))


def main():
    ap = argparse.ArgumentParser(description="RCT-side positive control")
    ap.add_argument("--cohorts", nargs="+", default=["act", "cogte", "breathe"])
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--readout", default="roi_norm")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--tlen", type=int, default=80)
    ap.add_argument("--out", default=str(RUNS_DIR / "v2" / "rct_positive_control.json"))
    args = ap.parse_args()
    C.configure_backends()
    xyz = load_xyz(XYZ_PATH)
    model = load_brainlm_forpretraining(Path(args.pretrained)).to(DEVICE).eval()
    rows = []
    from rct_probe import upsample_short

    for name in args.cohorts:
        co = load_cohort(name, require_labels=False)
        # Short cohorts must be brought up to --tlen before the encoder sees them, with the
        # LOCKED fill (zero-pad after per-ROI mean-centring; never stretch or tile).
        # Without this the script only works at a tlen every cohort natively exceeds, which
        # is why every published positive-control JSON in runs/v2 carries the --tlen 80
        # default: CogTE holds 95-118 frames, so at the FM's official 200 the readout raised
        # `shape '[8, 424, -1, 20]' is invalid for input of size 400256` (= 118 frames).
        # tlen 80 is not a defensible input length for this model, so the fix is here rather
        # than in the default.
        if int(co.valid.min()) < args.tlen:
            co, _ = upsample_short(co, args.tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
        ids = list(co.subjects)
        ctl = CONTROLS[name](ids)
        x1 = co.bold[..., 0]
        feats = {
            "brainlm": brainlm_readout_batched(model, x1[:, :, :args.tlen].contiguous(),
                                               xyz, args.layer, args.readout),
            "fc_win": fc_feature(x1[:, :, :args.tlen].numpy(),
                                 np.full(len(x1), args.tlen)),
            "fc_full": fc_feature(x1.numpy(), co.valid[:, 0].numpy()),
            "amp": amp_feature(x1.numpy(), co.valid[:, 0].numpy()),
        }
        for tname, (y, mode) in ctl.items():
            ok = np.isfinite(y)
            if ok.sum() < 40:
                print(f"  {name}/{tname}: only {int(ok.sum())} available, skipped")
                continue
            chance = ("0.500 AUC" if mode == "bin" else
                      "0.333 bal-acc" if mode == "multi" else "0.000 rho")
            for k, F in feats.items():
                m, s = score(F[ok], y[ok], mode)
                rows.append({"cohort": name, "target": tname, "mode": mode,
                             "feature": k, "n": int(ok.sum()), "score": m, "sd": s})
                print(f"  {name:8s} {tname:5s} ({mode:5s}, chance {chance:14s}) "
                      f"{k:9s} n={int(ok.sum()):3d}  {m:+.4f} +/- {s:.4f}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "rows": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
