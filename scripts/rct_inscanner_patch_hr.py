#!/usr/bin/env python3
"""Within-scan heart-rate decoding with subject-grouped nested probes.

Within-scan centering removes additive scan-level offsets, not trait-state
interactions or all physiological confounding. Windows are clustered within
subjects; their count is not an independent-sample power calculation. Reliability
values are descriptive estimates, not a directly comparable Spearman ceiling.
Within-scan shuffling refits all supervised steps and assumes exchangeability
of windows; autocorrelation-sensitive nulls remain a separate sensitivity check.
Bootstrap intervals condition on fitted OOF predictions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from benchmark_evaluation import (PROTOCOL_VERSION, prepare_nested, predict_prepared,
                                  paired_bootstrap, permutation_tail)
from cohorts import center_rois, cohort_tr, load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from paths import OFFICIAL_BRAINLM_13M, XYZ_PATH  # noqa: E402
from probe_readouts import _encoder  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Window file and the within-scan reliability ceiling per cohort, from the table above.
#: `ceil` is the naive-SE value, `ceil_lo` the sqrt(2)-inflated one. Both are printed: a
#: score is not interpretable against an unstated bound.
COHORTS = {
    "psopt": dict(hr="data/rct/psopt/pSOPT_MRI_MeanHR_windows_27.5s.csv",
                  ceil=0.923, ceil_lo=0.846),
    # BREATHE keeps only 15 scans from 14 subjects once the analysis filters are applied,
    # and its ceiling is +0.621 (+0.242 conservative). Retained as a replication attempt,
    # NOT as a second test: a null here is uninformative at that n and that bound.
    "breathe": dict(hr="data/rct/breathe/BREATHE_MRI_MeanHR_windows_20.2s.csv",
                    ceil=0.621, ceil_lo=0.242),
}
#: The visits the imaging tensor carries. psopt's ECG tables also cover T3.
VISITS = ("T1", "T2")
PLAUSIBLE = (40.0, 120.0)
MIN_WINDOWS = 6
GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def subj_key(x):
    m = re.search(r"(\d+)", str(x))
    return m.group(1).lstrip("0") if m else None


@torch.no_grad()
def patch_readouts(model, x, xyz, layer, batch=8):
    """Per-temporal-patch readouts: {'FMmean': [N,P,hid], 'FMroi': [N,P,roi]}.

    The encoder's token grid is ROI-major, `[B, 1+roi*patch, hid]`, exactly as
    probe_readouts.extract reshapes it. Every published readout in this project then collapses
    the PATCH axis (`cls`, `mean`, `roi_norm`, `roi_var`, `emb_fc` all do). Here the ROI axis
    is collapsed instead and the patch axis is KEPT, which is what makes a within-scan
    question askable at all.
    """
    n_patch = x.shape[2] // C.TIMEPOINT_PATCHING_SIZE
    n_roi = x.shape[1]
    out = {"FMmean": [], "FMroi": []}
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(DEVICE)
        h = _encoder(model)(signal_vectors=xb,
                            xyz_vectors=xyz.unsqueeze(0).repeat(xb.shape[0], 1, 1).to(DEVICE),
                            noise=None, return_dict=True,
                            output_hidden_states=True).hidden_states[layer]
        g = h[:, 1:, :].reshape(h.shape[0], n_roi, n_patch, h.shape[-1])
        out["FMmean"].append(g.mean(dim=1).float().cpu())              # [B, P, hid]
        out["FMroi"].append(g.norm(dim=-1).transpose(1, 2).float().cpu())  # [B, P, roi]
        print(f"  {min(s + batch, len(x))}/{len(x)}", end="\r", flush=True)
    print()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def patch_amp(x):
    """Per-ROI sd inside each temporal patch: [N, P, roi]. The closed-form control."""
    n = C.TIMEPOINT_PATCHING_SIZE
    P = x.shape[2] // n
    a = x[:, :, :P * n].reshape(x.shape[0], x.shape[1], P, n)
    return a.std(dim=3, unbiased=False).transpose(1, 2).numpy()


def oof_cluster(blocks, ks, y, groups, seed, n_splits=5):
    # Folds independent of HR, reusable for the full-pipeline permutation null.
    prep = prepare_nested(blocks, ks, np.zeros(len(y)), groups, seed, n_splits)
    return predict_prepared(prep, y, binary=False)


def sc(y, p):
    r = stats.spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else float("nan")


def cluster_boot(y, pa, pb, groups, n_boot, seed=0):
    return paired_bootstrap(y, pa, pb, n_boot, seed, groups=groups)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohorts", nargs="+", default=["psopt", "breathe"])
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--release", default="amp", choices=("amp", "zscore", "schaefer", "amp_nogsr", "zscore_nogsr", "schaefer_nogsr"),
                    help="Which stored file to read. The *_nogsr variants exist only for psopt, "
                         "which ships the same recordings under two denoisings "
                         "(pSOPT_visits12_GSR / pSOPT_visits12_noGSR, delivered "
                         "2026-09-17). Default `amp` is the GSR build, matching what "
                         "act/cogte/breathe's A424 releases are.")
    ap.add_argument("--out", default="runs/benchmark_v1/rct_inscanner_patch_hr.json")
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen

    from model_loader import load_brainlm_forpretraining

    xyz = load_xyz(XYZ_PATH)
    model = load_brainlm_forpretraining(Path(a.pretrained)).to(DEVICE).eval()
    n_patch = a.tlen // C.TIMEPOINT_PATCHING_SIZE
    rows, tests = [], []

    for cname in a.cohorts:
        spec = COHORTS[cname]
        tr = cohort_tr(cname)
        print(f"\n{'=' * 90}\n{cname}  TR {tr}  tlen {a.tlen}  "
              f"{n_patch} patches of {C.TIMEPOINT_PATCHING_SIZE * tr:.1f}s\n{'=' * 90}", flush=True)

        co = center_rois(load_cohort(cname, release=a.release, require_labels=False))
        if int(co.valid.min()) < a.tlen:
            raise SystemExit(f"{cname}: shortest session {int(co.valid.min())} < tlen {a.tlen}")
        keys = np.array([subj_key(s) for s in co.subjects])

        # --- target: within-scan-centred MeanHR inside the analysis window ---------------
        w = pd.read_csv(spec["hr"])
        if "Run" in w.columns:
            w = w[w.Run == "rest-run01"]
        w["k"] = w.Subject.map(subj_key)
        w = w[(w.Usable_HR) & w.MeanHR.between(*PLAUSIBLE)
              & (w.fMRIWindowEndSec <= C.TIMEPOINT_PATCHING_SIZE * tr * n_patch + 0.5)
              & (w.WindowIdx < n_patch)]
        # The imaging tensor holds T1 and T2 only; psopt's ECG file also carries T3 (450 of
        # its 1420 rows), which has no imaging here. Dropping it BEFORE the per-scan count is
        # safe and not a choice about power: T3 forms its own (k, Timepoint) group, so no
        # surviving T1/T2 group's window count changes. Restricting to imaged subjects for
        # the same reason. Without this the loop below hit `KeyError: 'T3'`.
        w = w[w.Timepoint.isin(VISITS) & w.k.isin(set(keys[keys != None]))]
        keep = w.groupby(["k", "Timepoint"]).MeanHR.transform("size") >= MIN_WINDOWS
        w = w[keep].copy()
        w["dev"] = w.MeanHR - w.groupby(["k", "Timepoint"]).MeanHR.transform("mean")
        print(f"target: {len(w)} windows / {w.groupby(['k', 'Timepoint']).ngroups} scans / "
              f"{w.k.nunique()} subjects   within-scan dev SD {w.dev.std():.2f} bpm", flush=True)

        # --- features, per patch, per session --------------------------------------------
        feats = {}
        for ti, tp in enumerate(("T1", "T2")):
            x = co.bold[..., ti][:, :, :a.tlen].contiguous()
            f = patch_readouts(model, x, xyz, a.layer)
            f["amp"] = patch_amp(x)
            feats[tp] = f
            print(f"  {tp}: " + "  ".join(f"{k}{v.shape}" for k, v in f.items()), flush=True)

        # --- assemble the row table: one row per (scan, patch) ---------------------------
        sidx = {kk: i for i, kk in enumerate(keys)}
        R = {nm: [] for nm in ("FMmean", "FMroi", "amp")}
        yv, grp, pos = [], [], []
        for (kk, tp), d in w.groupby(["k", "Timepoint"]):
            if kk not in sidx or tp not in feats:
                continue
            i = sidx[kk]
            pid = d.WindowIdx.values.astype(int)
            for nm in R:
                blk = feats[tp][nm][i][pid]                       # [n_win, dim]
                R[nm].append(blk - blk.mean(axis=0, keepdims=True))   # within-scan centred
            yv.append(d.dev.values)
            grp.append(np.full(len(d), kk))
            pos.append(pid)
        if not yv:
            print(f"  {cname}: no overlap between imaging and ECG windows, skipped")
            continue
        y = np.concatenate(yv)
        groups = np.concatenate(grp)
        pos = np.concatenate(pos)
        F = {nm: np.vstack(v) for nm, v in R.items()}
        # `pos` as one-hot, centred within scan exactly like every other block, so the
        # comparison is between blocks and not between preprocessing choices.
        oh = np.zeros((len(pos), n_patch))
        oh[np.arange(len(pos)), pos] = 1.0
        ohc = np.vstack([b - b.mean(axis=0, keepdims=True)
                         for b in np.split(oh, np.cumsum([len(v) for v in yv])[:-1])])
        F["pos"] = ohc
        print(f"rows {len(y)}  subjects {len(np.unique(groups))}  "
              f"blocks " + "  ".join(f"{k}:{v.shape[1]}" for k, v in F.items()), flush=True)

        # --- arms -------------------------------------------------------------------------
        arms = {"pos": ([F["pos"]], [n_patch])}
        for nm in ("amp", "FMmean", "FMroi"):
            arms[f"{nm}({a.k})"] = ([F[nm]], [a.k])
        oof, prepared = {}, {}
        print(f"\nceiling [within-scan reliability] {spec['ceil']:.3f}   "
              f"[SE x sqrt2] {spec['ceil_lo']:.3f}")
        for nm, (blocks, ks) in arms.items():
            prepared[nm] = [prepare_nested(blocks, ks, np.zeros(len(y)), groups, s)
                            for s in a.seeds]
            p = np.mean([predict_prepared(prep, y, binary=False)
                         for prep in prepared[nm]], axis=0)
            oof[nm] = p
            s = sc(y, p)
            rows.append(dict(cohort=cname, arm=nm, n_rows=int(len(y)),
                             n_subj=int(len(np.unique(groups))), score=s,
                             ceil=spec["ceil"], ceil_lo=spec["ceil_lo"]))
            print(f"  {nm:14s} rho {s:+.4f}   {100 * s / spec['ceil']:6.1f}% of ceiling",
                  flush=True)

        # --- within-scan permutation null for the FM arm ----------------------------------
        cuts = np.cumsum([len(v) for v in yv])[:-1]
        rng = np.random.default_rng(0)
        null = []
        for _ in range(a.n_perm):
            yp = np.concatenate([rng.permutation(b) for b in np.split(y, cuts)])
            prediction = np.mean([predict_prepared(prep, yp, binary=False)
                                  for prep in prepared[f"FMmean({a.k})"]], axis=0)
            null.append(sc(yp, prediction))
        null = np.array(null)
        obs = sc(y, oof[f"FMmean({a.k})"])
        pperm = permutation_tail(obs, null)
        print(f"  within-scan permutation null: mean {null.mean():+.4f} "
              f"sd {null.std():.4f}  p(FMmean) {pperm:.4f}", flush=True)
        rows.append(dict(cohort=cname, arm="FMmean_perm_null", n_rows=int(len(y)),
                         score=float(null.mean()), sd=float(null.std()), p=pperm,
                         ceil=spec["ceil"], ceil_lo=spec["ceil_lo"]))

        # --- the pairs that decide it ------------------------------------------------------
        PAIRS = [("pos", f"FMmean({a.k})", "FM vs patch index: clears the positional floor?"),
                 (f"amp({a.k})", f"FMmean({a.k})",
                  "THE key test: amp is the closed-form read of cardiac amplitude modulation"),
                 (f"amp({a.k})", f"FMroi({a.k})", "same test for the per-patch ROI map"),
                 ("pos", f"amp({a.k})", "does amp itself clear the positional floor?")]
        for u, v, note in PAIRS:
            if u not in oof or v not in oof:
                continue
            d, p, ci, detail = paired_bootstrap(y, oof[u], oof[v], a.n_boot,
                                                groups=groups, return_details=True)
            tests.append(dict(cohort=cname, a=u, b=v, note=note, delta=d, p=p, **detail,
                              ci_lo=ci[0], ci_hi=ci[1], n_rows=int(len(y)),
                              n_subj=int(len(np.unique(groups)))))
            print(f"  {v:14s} vs {u:14s} delta {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
                  f" p {p:.4f}{' *' if p < 0.05 else ''}\n      {note}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, "args": vars(a), "scores": rows, "tests": tests},
                                      indent=1))
    print("\n" + "=" * 90)
    print("HOW TO READ THIS")
    print("=" * 90)
    print("  Within-scan centering removes additive scan offsets; trait-state interactions")
    print("  and other time-varying confounding can remain. CIs condition on fitted OOF models.")
    print("  The FM earns credit only by beating BOTH `pos` (it has positional embeddings,")
    print("  so it gets patch index for free) AND `amp` at matched k. Intervals are cluster")
    print("  bootstraps over SUBJECTS -- a subject contributes up to 20 dependent rows.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
