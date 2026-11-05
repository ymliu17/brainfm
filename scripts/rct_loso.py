#!/usr/bin/env python3
"""Leave-one-study-out evaluation with source-only preprocessing by default.

--harmonization transductive explicitly permits held-out cohort feature statistics.
Tucker features always share one training-study basis across every cohort/visit.
Target scales are standardized within source cohorts only. Held-out labels are
used only for scoring and fixed-model permutation inference, never model choice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from benchmark_evaluation import PROTOCOL_VERSION, permutation_tail
from paths import (BRAINLM_111M, BRAINLM_650M, OFFICIAL_BRAINLM_13M,  # noqa: E402
                   RUNS_DIR, XYZ_PATH, XYZ_SCHAEFER450)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#: Default is the published three. pSOPT can be added with `--cohorts act cogte breathe
#: psopt`, which turns this into a 4-fold leave-one-STUDY-out and is a real strengthening:
#: psopt is a different POPULATION (a late-life depression/apathy trial, MoCA 23.7, GDS-30
#: 5.7) rather than a fourth MCI cognitive-training trial, so a held-out psopt fold asks
#: whether an axis generalises across indication and not just across site.
#:
#: CAVEAT THAT MUST BE STATED WITH ANY 4-COHORT NUMBER: the within-cohort z-scoring this
#: script already applies harmonises SCALE, which is what psopt's raw-intensity release
#: differs by (median per-ROI sd 6.684 against 0.679-0.733). It does NOT harmonise
#: DENOISING, and psopt's A424 retains its global signal (GS/per-ROI-sd 0.711 against
#: 0.152/0.248/0.152). So run it both ways -- see scripts/mci_axis_transfer.py --gsr.
COH = ["act", "cogte", "breathe"]

#: label -> (kind, ckpt, readout, layer, atlas, tlen). Each model at its own last layer and
#: native parcellation, matching every other table in this project.
#: Supplied without model code; src/fmrilm.py loads it through the authors' repo clone.
FMRILM_CKPT = Path(__file__).resolve().parents[1] / "checkpoints/pretrained/fmri-lm/ckpt-best.pt"

FAMILIES = {
    "FC [A424-200]": ("fc", "", None, None, "amp", 200),
    "tucker-FC [A424-200]": ("tucker", "", None, None, "amp", 200),
    "amp [A424-200]": ("amp", "", None, None, "amp", 200),
    "BrainLM 13M [A424-200]": ("brainlm", str(OFFICIAL_BRAINLM_13M), "roi_norm", 4,
                               "amp", 200),
    "BrainLM 111M [A424-200]": ("vitmae", str(BRAINLM_111M.parent), "cls", None,
                                "amp", 200),
    "BrainLM 650M [A424-200]": ("vitmae", str(BRAINLM_650M.parent), "cls", None,
                                "amp", 200),
    "FC [Sch160]": ("fc", "", None, None, "schaefer", 160),
    "tucker-FC [Sch160]": ("tucker", "", None, None, "schaefer", 160),
    "amp [Sch160]": ("amp", "", None, None, "schaefer", 160),
    "Brain-JEPA [Sch160]": ("brainjepa", "", "mean", None, "schaefer", 160),
    # fMRI-LM stage-1 VQ tokenizer. Evaluated ONLY here, never on ADNI: arXiv 2511.21760
    # 4.1 says stages 1-2 use UKB and ABCD, but the checkpoint does not record its
    # --dataset_dir, and ADNI with target `AD` sits in the authors' own downstream suite
    # either way. Leave-one-study-out needs no ADNI, so it is the one contamination-free
    # comparison available. Both cells are fixed a priori -- the last block, which is the
    # only layer choice available without an ADNI sweep to pick one, with `mean` matching
    # Brain-JEPA's convention and `emb_fc` matching the readout that dominates for the 13M.
    "fMRI-LM mean [Sch160]": ("fmrilm", str(FMRILM_CKPT), "mean", 12, "schaefer", 160),
    "fMRI-LM emb_fc [Sch160]": ("fmrilm", str(FMRILM_CKPT), "emb_fc", 12, "schaefer", 160),
    "fMRI-LM rand-init [Sch160]": ("fmrilm-rand", str(FMRILM_CKPT), "emb_fc", 12,
                                   "schaefer", 160),
}

#: Families excluded from the default sweep. fMRI-LM is registered but opt-in: its
#: released stage-1 checkpoint reconstructs its own input at r = +0.0001 and uses 14
#: of 8192 codebook entries against 3485 for a random-init model of the same
#: architecture, so any number it produces describes the file, not the method.
#: Name it explicitly with --families to run it.
OPT_IN = {k for k in FAMILIES if "fMRI-LM" in k}
DEFAULT_FAMILIES = [k for k in FAMILIES if k not in OPT_IN]

#: (target, metric, which timepoint's features). The controls come first so a reader meets
#: them before the outcomes.
TARGETS = [("demo_sex", "auc", 0), ("demo_age", "rho", 0),
           ("mem_level", "rho", 0), ("mem_level_t2", "rho", 1),
           ("mem_change", "rho", 0), ("ef_change", "rho", 0)]


def zwithin(F):
    """Standardise a cohort's features with its OWN statistics. No labels involved."""
    sc = StandardScaler().fit(F)
    return sc.transform(F)


def fit_predict(Ftr, ytr, Fte, binary, n_pca=100):
    """Train on the pooled two cohorts, predict the held-out one."""
    n = min(n_pca, len(Ftr) - 1)
    if Ftr.shape[1] > n:
        p = PCA(n_components=n, random_state=0).fit(Ftr)
        Ftr, Fte = p.transform(Ftr), p.transform(Fte)
    if binary:
        m = LogisticRegression(C=0.01, max_iter=5000).fit(Ftr, ytr.astype(int))
        return m.predict_proba(Fte)[:, 1]
    return Ridge(alpha=1.0).fit(Ftr, ytr).predict(Fte)


def metric(y, p, kind):
    if kind == "auc":
        try:
            return float(roc_auc_score(y.astype(int), p))
        except ValueError:
            return float("nan")
    r = stats.spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else float("nan")


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES,
                    help="opt-in families are excluded by default; see OPT_IN")
    ap.add_argument("--cohorts", nargs="+", default=COH)
    ap.add_argument("--gsr", action="store_true",
                    help="Regress out each session's global signal first. The within-cohort z-scoring below harmonises SCALE but not DENOISING, and the four A424 releases differ on that axis (GS/per-ROI-sd 0.152/0.248/0.152 for act/cogte/breathe against 0.711 for psopt). Shares one implementation with mci_axis_transfer.apply_gsr. Default off, so the published invocation is unchanged.")
    ap.add_argument("--harmonization", choices=("source", "transductive"), default="source",
                    help="source: fit feature scaling on training cohorts only; transductive: use each cohort own unlabeled statistics")
    ap.add_argument("--n_pca", type=int, default=100)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--out", default=str(RUNS_DIR / "benchmark_v1/rct_loso.json"))
    a = ap.parse_args()
    C.configure_backends()
    import os
    os.chdir(ROOT)
    from cohorts import center_rois, load_cohort
    from rct_positive_control import CONTROLS
    from rct_probe import features, upsample_short
    from thirdorder import shared_coskew_basis

    rows = []
    for family_index, fam in enumerate(a.families):
        kind, ckpt, readout, layer, rel, tlen = FAMILIES[fam]
        C.NUM_TIMEPOINTS_PER_VOXEL = tlen
        model = None
        if kind not in {"fc", "amp", "tucker"}: gpu_acquire()
        if kind == "brainlm":
            from model_loader import load_brainlm_forpretraining
            model = load_brainlm_forpretraining(Path(ckpt)).to(DEVICE).eval()
        elif kind == "vitmae":
            from vitmae_input import load_vitmae
            model = load_vitmae(Path(ckpt), device=DEVICE)[0]
        elif kind == "brainjepa":
            from brainjepa import load_brainjepa
            model = load_brainjepa(device=DEVICE, verbose=False)
        elif kind in ("fmrilm", "fmrilm-rand"):
            from fmrilm import load_fmrilm
            model = load_fmrilm(Path(ckpt), device=DEVICE,
                                random_init=(kind == "fmrilm-rand"))
        xyz = load_xyz(XYZ_SCHAEFER450 if rel == "schaefer" else XYZ_PATH)

        F, Y, M, raw = {}, {}, {}, {}
        tucker_folds = {}
        for n in a.cohorts:
            co = center_rois(load_cohort(n, release=rel, require_labels=True))
            if a.gsr:
                from mci_axis_transfer import apply_gsr
                co = apply_gsr(co, n)
            if int(co.valid.min()) < tlen:
                co, _ = upsample_short(co, tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
            for tp in (0, 1):
                x = co.bold[..., tp]
                if kind == "tucker":
                    raw[(n, tp)] = x[:, :, :tlen].numpy()
                else:
                    F[(n, tp)] = features(kind, x, None, model, xyz, readout, layer,
                                          tlen, 64, None, tlen)
            ctrl = CONTROLS[n](co.subjects)
            Y[(n, "demo_sex")] = np.asarray(ctrl["sex"][0], float)
            Y[(n, "demo_age")] = np.asarray(ctrl["age"][0], float)
            Y[(n, "mem_level")] = np.asarray(co.baseline_levels("mem"), float)
            Y[(n, "mem_level_t2")] = np.asarray(co.followup_levels("mem"), float)
            Y[(n, "mem_change")] = np.asarray(co.change_scores("mem"), float)
            Y[(n, "ef_change")] = np.asarray(co.change_scores("ef"), float)
            M[(n, "demo")] = co.analysis_mask()
            M[(n, "mem")] = co.labelled_mask("mem")
            M[(n, "ef")] = co.labelled_mask("ef")
            print(f"  {fam}/{n}: scans={len(co)}", flush=True)
        del model
        torch.cuda.empty_cache()
        gpu_release()
        from benchmark_gpu_phase import has_future_gpu_work, release_gpu_for_cpu_tail
        if not has_future_gpu_work(FAMILIES[f][0] for f in a.families[family_index + 1:]):
            release_gpu_for_cpu_tail()

        for tgt, met, tp in TARGETS:
            grp = "demo" if tgt.startswith("demo") else tgt.split("_")[0]
            binary = met == "auc"
            for held in a.cohorts:
                tr = [c for c in a.cohorts if c != held]
                if kind == "tucker":
                    # One basis per held-out study, shared across cohorts AND visits.
                    if held not in tucker_folds:
                        from thirdorder import transfer_features
                        source = np.concatenate([raw[(c, 0)][M[(c, "demo")]] for c in tr])
                        keys = [(c, t) for c in a.cohorts for t in (0, 1)]
                        _, blocks = transfer_features(source, [raw[key] for key in keys])
                        tucker_folds[held] = dict(zip(keys, blocks))
                    F = tucker_folds[held]
                Xtr, ytr = [], []
                ok = True
                for c in tr:
                    m = M[(c, grp)] & np.isfinite(Y[(c, tgt)])
                    if m.sum() < 20:
                        ok = False
                        break
                    block = F[(c, tp)][m]
                    Xtr.append(zwithin(block) if a.harmonization == "transductive" else block)
                    yv = Y[(c, tgt)][m]
                    # z-score the OUTCOME within cohort too: the composites are on
                    # different scales (ACT memory sd 11.78, BREATHE 0.99), so pooling the
                    # raw values would make the regression fit a cohort indicator.
                    ytr.append(yv if binary else (yv - yv.mean()) / max(yv.std(), 1e-12))
                mh = M[(held, grp)] & np.isfinite(Y[(held, tgt)])
                if not ok or mh.sum() < 20:
                    continue
                Xtr = np.vstack(Xtr)
                ytr = np.concatenate(ytr)
                if binary and len(set(ytr.astype(int))) < 2:
                    continue
                Fte = F[(held, tp)][mh]
                if a.harmonization == "transductive":
                    Fte = zwithin(Fte)
                else:
                    scaler = StandardScaler().fit(Xtr)
                    Xtr, Fte = scaler.transform(Xtr), scaler.transform(Fte)
                yte = Y[(held, tgt)][mh]
                pred = fit_predict(Xtr, ytr, Fte, binary, a.n_pca)
                obs = metric(yte, pred, met)
                rr = np.random.default_rng(0)
                nul = np.array([metric(rr.permutation(yte), pred, met)
                                for _ in range(a.n_perm)])
                p = permutation_tail(obs, nul, two_sided=True,
                                     center=0.5 if binary else 0.0)
                rows.append({"family": fam, "target": tgt, "metric": met,
                             "held_out": held, "n_train": int(len(ytr)),
                             "n_test": int(mh.sum()), "score": obs, "p_perm": p,
                             "null_mean": float(np.nanmean(nul)),
                             "null_sd": float(np.nanstd(nul))})
                print(f"    {fam:<24}{tgt:<14}hold {held:<8}"
                      f"n_tr={len(ytr):3d} n_te={int(mh.sum()):3d} "
                      f"{met} {obs:+.4f}  p {p:.3f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"protocol_version": PROTOCOL_VERSION,
                                          "args": vars(a), "rows": rows}, indent=2))
    print(f"\n-> {a.out}  ({len(rows)} folds)")


if __name__ == "__main__":
    main()
