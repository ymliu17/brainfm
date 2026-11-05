#!/usr/bin/env python3
"""RCT-side probing: baseline cognitive LEVEL, continuous CHANGE, and the binary responder.

WHY THESE THREE TARGETS TOGETHER
--------------------------------
The paper's only downstream target is a binary responder label thresholded at change >= 0.
Three things are wrong with reading that alone, and all three are measurable rather than
rhetorical:

1. POWER. A change score is a difference of two noisy measurements, so it carries roughly
   twice the measurement-error variance of either one. This project measured the
   consequence: the pooled displacement test needs n ~ 1086 to detect the effect it
   observed, against 274 available, and 77.6% of the RCT's scan-to-scan delta variance is
   within-scan measurement noise. The BASELINE LEVEL is the same cognitive construct
   without the differencing penalty. If a representation cannot predict the level at
   n = 95/74/107, its null on the change carries no information about the model.

2. THRESHOLDING. `change >= 0` discards magnitude and puts 167 of 274 subjects in the
   positive class. `--exclude_near_zero` drops the subjects whose |change| is smallest,
   where the label is most likely to be measurement noise (Reviewer 2, M4a).

3. THE MAJORITY CLASS. On CogTE the majority class alone scores 76.3% accuracy against a
   published headline of 80.3%. Accuracy without that reference number is uninterpretable,
   so every binary result here is printed next to it.

PERMUTATION NULL
----------------
The observed score and its null come from the SAME pipeline: PCA reduction, then k-fold CV
with the regularisation strength chosen on an inner split of each training fold. Only the
target vector is shuffled. PCA is fitted once on all rows and shared between the observed
run and the null -- that can inflate both equally, but it cannot manufacture a difference
between them, which is what the p-value is about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from benchmark_evaluation import PROTOCOL_VERSION, permutation_tail
from cohorts import center_rois, load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from temporal import PATCH, resample_to_length  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import (OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH,  # noqa: E402
                   XYZ_SCHAEFER450)
from protocol import compute_metrics, majority_baseline  # noqa: E402
from thirdorder import shared_coskew_basis, tucker_fc_feature  # noqa: E402
from probe_readouts import brainlm_readout_batched  # noqa: E402
from transfer_baseline import amp_feature, fc_feature  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def upsample_short(co, need, pad_tol, fill="auto"):
    """Bring every scan in a cohort up to `need` frames, the way prepare_adni.py does.

    Both encoders were pretrained at a fixed sequence length -- BrainLM on 200 frames
    (10 tokens of 20) and Brain-JEPA on 160 (10 patches of 16) -- and ACT and BREATHE
    supply those natively (291-300 valid frames). CogTE does not: 95-118 valid frames at
    TR 2.5 s. Truncating everyone to CogTE's capacity was what forced the whole table down
    to 80 frames, and the two models do NOT pay the same price for that. Measured against
    the pretrained length on 32 ACT subjects, BrainLM's between-subject distance geometry
    survives truncation (rho 0.92-0.97 at 4 of 10 tokens) because its positional encoding
    is sinusoidal, indexed by TIME position only (`pe[:x.size(2)]`), with ROI identity
    carried separately by the xyz projection. Brain-JEPA's is a learned 4500-row table
    joint over ROI and time, and it does not: rho 0.663 (mean) and 0.123 (roi_norm) at 6
    of 10 patches. A common-truncation table is therefore biased toward BrainLM by
    construction.

    Three paths, per scan, matching `prepare_adni.py --short_policy adaptive` so the two
    halves of the project resolve the same shortfall the same way:

        crop      the scan already has `need` valid frames
        pad       short by at most `pad_tol` (one patch)  -- zeros, after mean-centring
        stretch   short by more                           -- resampled to exactly `need`

    ACT and BREATHE take the crop path throughout. CogTE takes stretch: 95-118 valid
    frames against 160 or 200, i.e. 1.68x or 2.11x, which uses the whole recording and
    leaves every token real. Sending it to TR 0.735 s and cropping instead would read only
    the first 60% of each run.

    Per scan, because usable lengths differ within CogTE and a censored run must
    contribute only real frames.

    No z-scoring afterwards, unlike `datasets.apply_temporal_spec`: between-ROI amplitude
    is load-bearing here (the amplitude profile alone out-predicts the 13M encoder on
    ADNI CN/AD, 0.682 vs 0.630), so it is left intact.
    """
    v = co.valid.numpy()
    if int(v.min()) >= need:
        return co, "crop"
    out = torch.empty((co.bold.shape[0], co.bold.shape[1], need, co.bold.shape[3]),
                      dtype=torch.float32)
    paths = set()
    for tp in range(co.bold.shape[3]):
        for length in sorted(set(v[:, tp].tolist())):
            idx = np.nonzero(v[:, tp] == length)[0]
            blk = co.bold[idx][:, :, :int(length), tp].float()
            if length >= need:
                out[idx, :, :, tp] = blk[:, :, :need]
                paths.add("crop")
            elif fill in ("tile", "reflect"):
                # Repeat the recording instead of filtering it or inserting constants.
                # `reflect` time-reverses alternate copies so the seams are continuous.
                reps = -(-need // int(length))
                out[idx, :, :, tp] = torch.cat(
                    [blk if (fill == "tile" or k % 2 == 0) else torch.flip(blk, dims=[2])
                     for k in range(reps)], dim=2)[:, :, :need]
                paths.add(fill)
            elif fill == "pad" or (fill == "auto" and need - length <= pad_tol):
                blk = blk - blk.mean(dim=2, keepdim=True)
                out[idx, :, :, tp] = torch.cat(
                    [blk, blk.new_zeros(len(idx), blk.shape[1], need - int(length))], dim=2)
                paths.add("pad")
            else:
                out[idx, :, :, tp] = resample_to_length(blk, need, dim=2)
                paths.add("stretch")
    lo, hi = int(v.min()), int(v.max())
    detail = ""
    if "stretch" in paths:
        detail = f", x{need / hi:.2f}-x{need / lo:.2f} at TR {co.native_tr}"
    if "pad" in paths:
        detail += f", {need - hi}-{need - lo} frames padded ({(need - lo) / need:.0%} at worst)"
    print(f"  {co.name}: {lo}-{hi} valid frames -> {need} "
          f"({'+'.join(sorted(paths))}{detail}, amplitude preserved)", flush=True)
    return replace(co, bold=out, valid=torch.full_like(co.valid, need)), "+".join(sorted(paths))


def features(kind, x3, valid, model, xyz, readout, layer, tlen, rank, basis=None,
             jepa_nf=None):
    # BrainLM is always cropped to tlen: the encoder needs the token count ADNI used.
    # FC / amp / tucker have no such constraint, and how many frames they are given
    # changes them a lot -- ACT supplies 296 valid frames against the encoder's 80. Both
    # arms are worth reporting, so `valid` being non-None means "use every valid frame".
    if kind == "braingfm":
        from braingfm import pearson_fc, braingfm_fc_readout
        a = x3.numpy() if valid is not None else x3[:, :, :tlen].numpy()
        return braingfm_fc_readout(model, pearson_fc(a, valid))
    if kind == "brainlm":
        return brainlm_readout_batched(model, x3[:, :, :tlen].contiguous(), xyz, layer,
                                       readout)
    if kind == "vitmae":
        # BrainLM 111M / 650M. A token spans patch x patch ROIs and timepoints, so the
        # recording is padded into a square image and neighbouring ROIs are convolved
        # together -- which makes the y-order load-bearing here in a way it is not for the
        # 13M. At t140 roughly two thirds of the 650M's tokens are pure padding.
        from brainlm_input import roi_order_by_y
        from probe_readouts import extract_vitmae
        order = roi_order_by_y(XYZ_PATH)
        # The LAST layer, by design, for every family: `--layer` is not consulted here and
        # the value it records in the args block is inert. Left as-is rather than made to
        # honour the flag, because changing it would make every published number
        # irreproducible; the print makes the discrepancy visible instead.
        li = model.config.num_hidden_layers
        if layer is not None and int(layer) != li:
            print(f"    note: --layer {layer} ignored for kind=vitmae; using the last "
                  f"layer L{li} as designed", flush=True)
        out = extract_vitmae(model, model.config, x3[:, :, :tlen].contiguous(), order,
                             [li], [readout], batch=2)
        return np.asarray(out[(li, readout)])
    if kind == "fmrilm":
        # fMRI-LM stage-1 VQ tokenizer, same 450-ROI parcellation as Brain-JEPA. Its
        # patch_embed kernel is (1, 160), so 160 frames is not negotiable and short cohorts
        # arrive already zero-padded by the caller (the locked fill policy). The authors'
        # per-ROI robust normalisation is applied with constants estimated on the tensor in
        # hand, because `normalization_params.npz` is not distributed -- see src/fmrilm.py.
        from fmrilm import fmrilm_readout, robust_norm_params
        xin = x3[:, :, :tlen].contiguous()
        return fmrilm_readout(model, xin, readout=readout, layer=layer, batch=8,
                              norm=robust_norm_params(xin))
    if kind == "brainjepa":
        # 450 ROIs is baked into the released weights and is not a free parameter. The
        # frame count nominally is not either (pos_embed 4500 = 450 x 10 time patches), but
        # CogTE tops out at 95-118 valid frames, so `valid` is forwarded and jepa_readout
        # drops to a time sub-grid rather than padding 38% of the input with zeros.
        from brainjepa import jepa_readout
        return jepa_readout(model, x3, readout if readout in
                            ("mean", "roi_norm", "roi_mean") else "mean",
                            valid=np.full(len(x3), jepa_nf))
    if valid is not None:
        v, a = valid.astype(int), x3.numpy()
    else:
        v, a = np.full(len(x3), tlen), x3[:, :, :tlen].numpy()
    if kind == "fc":
        return fc_feature(a, v)
    if kind == "amp":
        return amp_feature(a, v)
    if kind == "tucker":
        # `basis` is fitted on this cohort's T1 scans and reused for T2, so the T2-minus-T1
        # delta is a difference in ONE coordinate system. Refitting per timepoint would
        # make the two sides of the delta incommensurable.
        return tucker_fc_feature(a, v, rank=rank, basis=basis)
    raise ValueError(kind)


def fold_transform(Atr, Ate, n_pca, n_pass=0):
    """Standardise then reduce, fitted on the training rows only.

    Order matters. The previous code reduced first (globally, over all rows) and then ran
    StandardScaler on the components, which rescales every component to unit variance.
    That flattens the eigenvalue spectrum, so the trailing components -- pure noise when
    n_pca is close to n -- get the same weight as the leading ones. Measured cost on ACT
    sex with 94 components from 95 samples: AUC 0.539 instead of 0.814.

    The last `n_pass` columns (cohort dummies) bypass the PCA and are appended unchanged.
    """
    from thirdorder import TuckerInput
    if isinstance(Atr, TuckerInput):
        Atr, Ate = Atr.fit_transform_pair(Ate)
    Ptr = Pte = None
    if n_pass:
        Atr, Ptr = Atr[:, :-n_pass], Atr[:, -n_pass:]
        Ate, Pte = Ate[:, :-n_pass], Ate[:, -n_pass:]
    sc = StandardScaler().fit(Atr)
    A, B = sc.transform(Atr), sc.transform(Ate)
    if A.shape[1] > n_pca:
        pc = PCA(n_components=min(n_pca, len(A) - 1), random_state=0).fit(A)
        A, B = pc.transform(A), pc.transform(B)
    if n_pass:
        A, B = np.hstack([A, Ptr]), np.hstack([B, Pte])
    return A, B


def _strat(y, binary):
    """Fold stratifier: the class for a binary target, quintile bins for a continuous one."""
    return (y.astype(int) if binary else
            np.clip(np.searchsorted(np.quantile(y, [0.2, 0.4, 0.6, 0.8]), y), 0, 4))


def prepare_folds(F, strat, seeds, n_splits, n_pca, n_pass):
    """Label-independent folds plus training-only scale->PCA transforms.

    The strat argument is retained for API compatibility but is not used. A fixed
    label-independent partition makes the observed and shuffled-label experiments
    identical except for y; rare single-class training folds predict their prior.


    The reduction is UNSUPERVISED -- it depends on the feature rows and the fold
    partition, not on y -- so it can be fitted once per (seed, fold) and reused for every
    regularisation strength and every label permutation.

    This is not just a speed-up. `permutation_p` used to call `cv_score` afresh per
    permutation, refitting the transform 200 x 5 x (6 x 3 + 1) = 19,000 times per cell on
    up to 89,676 dimensions; one sweep took 2.8 days. Holding the fold partition fixed
    across permutations is also the cleaner null: the CV split should not be a function of
    the permuted labels.
    """
    prep = []
    for seed in seeds:
        folds = []
        for tr, te in KFold(n_splits, shuffle=True,
                            random_state=seed).split(np.arange(len(F))):
            inner = [(itr, ite, fold_transform(F[tr][itr], F[tr][ite], n_pca, n_pass))
                     for itr, ite in KFold(3, shuffle=True, random_state=seed).split(np.arange(len(tr)))]
            folds.append((tr, te, fold_transform(F[tr], F[te], n_pca, n_pass), inner))
        prep.append(folds)
    return prep


def score_prepared(prep, y, binary):
    """Out-of-fold predictions over an already-prepared fold structure."""
    oofs = []
    for folds in prep:
        oof = np.full(len(y), np.nan)
        for tr, te, (Ao, Bo), inner in folds:
            best, best_s = GRID[0], -np.inf
            for c in GRID:
                p = np.full(len(tr), np.nan)
                for itr, ite, (Ai, Bi) in inner:
                    if binary and len(set(y[tr][itr])) < 2:
                        p[ite] = float(np.mean(y[tr][itr]))
                        continue
                    m = (LogisticRegression(C=c, max_iter=5000) if binary
                         else Ridge(alpha=1.0 / c)).fit(Ai, y[tr][itr])
                    p[ite] = (m.predict_proba(Bi)[:, 1] if binary else m.predict(Bi))
                ok = np.isfinite(p)
                if ok.sum() < 5:
                    continue
                s = _metric(y[tr][ok], p[ok], binary)
                if np.isfinite(s) and s > best_s:
                    best, best_s = c, s
            if binary and len(np.unique(y[tr])) < 2:
                oof[te] = float(np.mean(y[tr]))
                continue
            m = (LogisticRegression(C=best, max_iter=5000) if binary
                 else Ridge(alpha=1.0 / best)).fit(Ao, y[tr])
            oof[te] = (m.predict_proba(Bo)[:, 1] if binary else m.predict(Bo))
        oofs.append(oof)
    scores = [_metric(y, o, binary) for o in oofs]
    return float(np.mean(scores)), float(np.std(scores)), oofs


def cv_score(F, y, binary, seeds=(0, 1, 2), n_splits=5, n_pca=100, n_pass=0):
    """Pooled out-of-fold predictions across seeds. Returns (score, sd, oof_by_seed)."""
    prep = prepare_folds(F, _strat(y, binary), seeds, n_splits, n_pca, n_pass)
    return score_prepared(prep, y, binary)


def _metric(y, p, binary):
    if binary:
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y, p)
        except ValueError:
            return float("nan")
    r = stats.spearmanr(y, p).statistic
    return float(r) if np.isfinite(r) else float("nan")


def permutation_p(F, y, binary, obs, n_perm, seed=0, n_pca=100, n_pass=0,
                  seeds=(0, 1, 2), n_splits=5):
    """Label-permutation null, sharing ONE prepared fold structure across permutations."""
    prep = prepare_folds(F, _strat(y, binary), seeds, n_splits, n_pca, n_pass)
    expected = score_prepared(prep, y, binary)[0]
    if not np.isclose(obs, expected, atol=1e-8, rtol=1e-6):
        raise ValueError("Observed and null must use the same folds, seeds and statistic")
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = score_prepared(prep, rng.permutation(y), binary)[0]
    return permutation_tail(obs, null), float(np.mean(null)), float(np.std(null))


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description="RCT-side probe: level, change, responder")
    ap.add_argument("--kinds", nargs="+", default=["brainlm", "fc", "amp", "tucker"])
    ap.add_argument("--vitmae_dir",
                    default="checkpoints/pretrained/brainlm_vitmae_111M",
                    help="directory of a BrainLM ViTMAE release (111M or 650M), used by "
                         "--kinds vitmae. These stay on A424 like the 13M.")
    ap.add_argument("--jepa_ckpt",
                    default="checkpoints/pretrained/brainjepa/jepa-ep300.pth.tar")
    ap.add_argument("--jepa_gradient",
                    default="checkpoints/pretrained/brainjepa/gradient_mapping_450.csv")
    ap.add_argument("--release", default="amp", choices=("amp", "zscore", "schaefer", "amp_nogsr", "zscore_nogsr", "schaefer_nogsr"),
                    help="which imaging release to probe. `schaefer` is the 450-ROI "
                         "Schaefer-400 + Tian-S3 build that Brain-JEPA requires; it is NOT "
                         "interchangeable with the A424 builds (different denoising, worth "
                         "0.047 AUC on ADNI CN/AD), so FC/amp baselines have to be "
                         "recomputed on it rather than borrowed from the A424 runs. "
                         "--kinds brainjepa requires it.")
    ap.add_argument("--targets", nargs="+",
                    default=["mem_level", "ef_level", "mem_change", "ef_change",
                             "mem_bi", "ef_bi"])
    ap.add_argument("--cohorts", nargs="+", default=["act", "cogte", "breathe"])
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--readout", default="roi_norm")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--input", default="t1", choices=("t1", "t2", "t1t2", "delta"),
                    help="t1 = baseline scan only (the only honest option for a BASELINE "
                         "level target); t2 = the follow-up scan alone, whose matching "
                         "target is *_level_t2; t1t2 = both scans concatenated; "
                         "delta = T2 - T1. Pairing `--input t1 --targets mem_level` with "
                         "`--input t2 --targets mem_level_t2` measures the same "
                         "cross-sectional question twice in the same subjects, which is "
                         "the internal replication the cross-sectional arm lacked.")
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--resample_short", action=argparse.BooleanOptionalAction, default=True,
                    help="bring every scan up to --tlen so each encoder runs at the "
                         "sequence length it was pretrained on (BrainLM 200 frames, "
                         "Brain-JEPA 160), instead of truncating every cohort to what the "
                         "shortest one supplies. Same three paths as prepare_adni.py "
                         "--short_policy adaptive: crop if long enough, pad if short by at "
                         "most --pad_tolerance, stretch to exactly --tlen otherwise. Only "
                         "CogTE is short (95-118 valid frames); ACT and BREATHE supply "
                         "291+ and are untouched. See upsample_short() for why a common "
                         "truncation instead biases the table toward BrainLM.")
    ap.add_argument("--pad_tolerance", type=int, default=None,
                    help="largest shortfall zero-padded rather than stretched; defaults to "
                         "one temporal patch (20).")
    ap.add_argument("--fill", default="pad",
                    choices=("auto", "pad", "stretch", "tile", "reflect"),
                    help="LOCKED 2026-09-01: a scan short of --tlen is ZERO-PADDED, after per-ROI mean-centring. Not stretched, not tiled. Decided by measurement across four comparisons -- pad beat stretch 24/25 and 26/33 cells, beat tile 19/25 on ADNI and 5/9 on CogTE, beat reflect 23/25 and 9/9 -- covering shortfalls from 30%% (ADNI 140->200) to 53%% (CogTE 95->200). Stretching filters every acquired sample; tiling injects a periodicity the recording never had, which costs the ENCODERS specifically (on CogTE all six FC/amp cells moved under 0.01 while BrainLM age fell 0.134). The other choices remain available for re-testing a genuinely new regime; they are not the default and should not be made one without numbers. "
                         "Values other than the default: `auto` restores the old tolerance "
                         "rule. "
                         "`auto` keeps the tolerance rule (pad small shortfalls, stretch "
                         "large ones). `pad` or `stretch` forces one for every shortfall, "
                         "which is what the A/B in scripts/shortfall_ab.py selects between: "
                         "CogTE needs 41-53%% of its sequence filled, far outside the 30%% "
                         "the ADNI comparison covered, so the choice there is made on its "
                         "own measurement rather than extrapolated.")
    ap.add_argument("--full_length", action="store_true",
                    help="give FC/amp/tucker every valid frame instead of tlen. BrainLM is "
                         "cropped either way, so this arm answers 'is the baseline winning "
                         "because of the representation or because it sees 3.7x the data'.")
    ap.add_argument("--tucker_rank", type=int, default=64)
    ap.add_argument("--n_pca", type=int, default=100)
    ap.add_argument("--exclude_near_zero", type=float, default=0.0,
                    help="drop subjects whose |change| falls below this quantile of "
                         "|change| within the cohort (Reviewer 2 M4a sensitivity check)")
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--no_center_rois", dest="center_rois", action="store_false",
                    help="skip per-ROI mean-centring of the RCT scans. Only for "
                         "reproducing pre-2026-09-02 numbers: the ADNI build the axis "
                         "comes from IS centred, so leaving the RCT side un-centred is a "
                         "distribution mismatch on the transfer path, worth up to 0.24 AUC "
                         "on the decodability control.")
    ap.add_argument("--pooled", action="store_true",
                    help="ALSO fit one model on all cohorts at once (n=276 instead of "
                         "~92), with cohort entered as dummy covariates and the target "
                         "z-scored within cohort. Motivated by the only non-null result "
                         "found so far being a CROSS-cohort transfer (CogTE<->BREATHE "
                         "mem_bi, perm p 0.003/0.007) while every within-cohort probe was "
                         "at chance -- if information exists only across cohorts, pooling "
                         "is the natural test. Folds are grouped so no cohort is split "
                         "across train and test in a way that lets the dummy leak.")
    ap.add_argument("--shuffle_subjects", type=int, default=0,
                    help="non-zero: permute feature rows within each cohort with this "
                         "seed, producing one draw of the table-level null instead of a "
                         "result. Use scripts/subject_shuffle_null.py to drive it.")
    ap.add_argument("--cross_cohort", action="store_true",
                    help="also fit on one cohort and score every other cohort")
    ap.add_argument("--out", default=str(RUNS_DIR / "benchmark_v1" / "rct_probe.json"))
    args = ap.parse_args()
    if "braingfm" in args.kinds:
        ap.error("BrainGFM requires its checkpoint atlas, unavailable in A424/Sch450 releases; use scripts/rct_braingfm.py with verified native-atlas FC files")
    if "tucker" in args.kinds and (args.pooled or args.cross_cohort):
        ap.error("Tucker pooled/pairwise transfer needs a source-shared basis; use rct_loso.py with a listed tucker-FC family for cross-study evaluation")
    C.configure_backends()

    # BrainLM's ROI count comes from the xyz table, not from the weights, so the
    # coordinate file follows the release.
    xyz = load_xyz(XYZ_SCHAEFER450 if args.release == "schaefer" else XYZ_PATH)
    gpu_acquire() if any(k not in {"fc", "amp", "tucker"} for k in args.kinds) else None
    model = (load_brainlm_forpretraining(Path(args.pretrained)).to(DEVICE).eval()
             if "brainlm" in args.kinds else None)
    if "vitmae" in args.kinds:
        from vitmae_input import load_vitmae
        model, _vcfg = load_vitmae(Path(args.vitmae_dir), device=DEVICE)
    if "brainjepa" in args.kinds:
        from brainjepa import load_brainjepa
        model = load_brainjepa(Path(args.jepa_ckpt), Path(args.jepa_gradient)).to(
            DEVICE).eval()

    if "vitmae" in args.kinds and args.release == "schaefer":
        raise SystemExit("--kinds vitmae stays on A424: the ViTMAE releases reorder ROIs by "
                         "y coordinate and tile neighbours into one patch, so moving them "
                         "to another parcellation is an unvalidated change. Compare them "
                         "against the A424 FC / tucker-FC baselines.")
    if "brainjepa" in args.kinds and args.release != "schaefer":
        raise SystemExit("--kinds brainjepa needs --release schaefer: the released weights "
                         "have a 4500-row position embedding (450 ROIs x 10 time patches) "
                         "and cannot consume A424's 424 ROIs.")
    gpu_release()
    coh = {n: load_cohort(n, release=args.release) for n in args.cohorts}
    if args.center_rois:
        # Match the ADNI build the transfer axis is fitted on, which is
        # `--amplitude official`: per-ROI mean-centring, no division. Without this the
        # encoders saw ACT and BREATHE with their per-ROI baseline offsets intact -- larger
        # than the signal in the A424 release -- while CogTE was centred by accident on its
        # way through `upsample_short`. See cohorts.center_rois for the measured effect.
        coh = {n: center_rois(c) for n, c in coh.items()}
        print("  per-ROI mean-centred (matches ADNI --amplitude official)", flush=True)
    if args.resample_short:
        tol = args.pad_tolerance if args.pad_tolerance is not None else PATCH
        moved = []
        for n in coh:
            coh[n], how = upsample_short(coh[n], args.tlen, tol, args.fill)
            if how != "crop":
                moved.append(f"{n}({how})")
        print(f"  brought up to {args.tlen} frames: "
              f"{', '.join(moved) if moved else 'none needed'}", flush=True)

    # Brain-JEPA reads a fixed 450 x 10 grid. CogTE tops out at 95 valid frames, so some
    # cohort must be cut -- and the cut has to be the SAME everywhere. Measured on 24 ACT
    # subjects, the full 10-patch grid and a 6-patch sub-grid give per-subject feature
    # vectors that correlate 0.983 (mean) yet reproduce the BETWEEN-subject distance
    # geometry at only rho 0.663 (mean) and 0.123 (roi_norm). Between-subject structure is
    # the entire signal a probe uses, so a mixed-grid run would train on one geometry and
    # test on another -- the cross-cohort rows would be meaningless. One grid for all.
    jepa_nf = None
    if "brainjepa" in args.kinds:
        vmin = int(min(int(co.valid.min()) for co in coh.values()))
        jepa_nf = min(160, vmin // 16 * 16)
        if jepa_nf < 16:
            raise SystemExit(f"shortest scan has {vmin} valid frames, under one 16-frame patch")
        print(f"brainjepa: uniform {jepa_nf} frames ({jepa_nf // 16}/10 time patches) across "
              f"{', '.join(coh)} -- set by the shortest scan ({vmin} valid frames)", flush=True)

    rows, cross_rows = [], []
    for kind_index, kind in enumerate(args.kinds):
        if kind not in {"fc", "amp", "tucker"}: gpu_acquire()
        # features per cohort, computed once and reused across all targets
        feats = {}
        for n, co in coh.items():
            v1 = co.valid[:, 0].numpy() if args.full_length else None
            v2 = co.valid[:, 1].numpy() if args.full_length else None
            basis = None
            if kind == "tucker":
                from thirdorder import TuckerInput
                scans = np.moveaxis(co.bold.numpy(), -1, 1)
                if not args.full_length:
                    scans = scans[..., :args.tlen]
                valid = np.minimum(co.valid.numpy(), scans.shape[-1])
                feats[n] = TuckerInput(scans, valid, args.tucker_rank, args.input)
                print(f"  tucker/{n}: deferred nested-fold features {feats[n].shape}", flush=True)
                continue
            f1 = features(kind, co.bold[..., 0], v1, model, xyz, args.readout,
                          args.layer, args.tlen, args.tucker_rank, basis, jepa_nf)
            if args.input == "t1":
                feats[n] = f1
            elif args.input == "t2":
                feats[n] = features(kind, co.bold[..., 1], v2, model, xyz, args.readout,
                                    args.layer, args.tlen, args.tucker_rank, basis, jepa_nf)
            else:
                f2 = features(kind, co.bold[..., 1], v2, model, xyz, args.readout,
                              args.layer, args.tlen, args.tucker_rank, basis, jepa_nf)
                feats[n] = (np.hstack([f1, f2]) if args.input == "t1t2" else f2 - f1)
            print(f"  {kind}/{n}: features {feats[n].shape}", flush=True)

        gpu_release()
        # Remaining FC/amplitude/Tucker arms and all probes are CPU-only.
        from benchmark_gpu_phase import has_future_gpu_work, release_gpu_for_cpu_tail
        if not has_future_gpu_work(args.kinds[kind_index + 1:]):
            model = None
            release_gpu_for_cpu_tail()

        if args.shuffle_subjects:
            # Aggregate null for the whole cell table. Permuting the FEATURE rows within a
            # cohort destroys the subject-to-outcome pairing while leaving the feature
            # distribution, the outcome vector and every analysis mask exactly as they
            # were. Applied here -- once, before any cell is evaluated -- so all cells in a
            # draw share one permutation and the draw inherits the same inter-cell
            # correlation as the real table. That is the point: per-cell p values cannot be
            # counted against 5% when the cells share subjects, nested targets and
            # overlapping features, so the null has to be built at the table level.
            # Masks come from `coh[...]`, i.e. the outcome side, and are deliberately NOT
            # permuted; permuting them too would merely relabel subjects and preserve the
            # pairing.
            for n in feats:
                rg = np.random.default_rng(abs(hash((args.shuffle_subjects, n))) % 2**32)
                feats[n] = feats[n][rg.permutation(len(feats[n]))]
            print(f"  SUBJECT-SHUFFLED (seed {args.shuffle_subjects}): feature rows "
                  f"permuted within each cohort; this run is a null draw, not a result",
                  flush=True)

        for tname in args.targets:
            # `arm_<contrast>` targets are the randomised-arm contrasts: the only NOISELESS
            # outcomes the trials have. They must be read on the T2-T1 change; on baseline
            # they are a negative control that has to come out null, because arm is
            # randomised and so cannot be predicted from a pre-intervention scan.
            # `demo_<var>` are the acquisition / demographic controls: age, sex,
            # acquisition site, and CogTE's cortical thickness. They are not outcomes --
            # they exist to say whether these imaging tensors carry ANY decodable signal,
            # which is what separates "this representation does not predict cognition"
            # from "this tensor is broken". They run through the SAME probe as everything
            # else (same folds, same nested C selection, same permutation null) so a
            # control number and an outcome number are directly comparable; the earlier
            # rct_positive_control.py had its own CV path and could not be read against
            # this table.
            is_demo = tname.startswith("demo_")
            is_arm = tname.startswith("arm_")
            if is_demo:
                label, kindname = "demo", tname[5:]
                binary = None                        # set per cohort from CONTROLS
            elif is_arm:
                label, kindname, binary = "mem", tname[4:], True
            else:
                label, kindname = tname.split("_", 1)
                binary = kindname == "bi"
            if is_arm and args.input == "t1":
                print(f"  NOTE {tname} on --input t1 is the NEGATIVE control "
                      f"(randomised arm vs pre-intervention scan); expect null.",
                      flush=True)
            per, pooled_F, pooled_y, pooled_c = {}, [], [], []
            for n, co in coh.items():
                if is_demo:
                    from rct_positive_control import CONTROLS
                    ctrl = CONTROLS[n](co.subjects)
                    if kindname not in ctrl:
                        print(f"  {n}/{tname}: not recorded for this cohort, skipped",
                              flush=True)
                        continue
                    y, mode = ctrl[kindname]
                    y = np.asarray(y, dtype=float)
                    if mode == "multi":
                        # Site is multi-class. One-vs-rest against the largest class keeps
                        # it on the AUC scale the rest of the table uses; the alternative
                        # (macro-averaged multiclass AUC) is not comparable to any other
                        # row here.
                        vals, cnt = np.unique(y[np.isfinite(y)], return_counts=True)
                        y = (y == vals[cnt.argmax()]).astype(float)
                    binary = mode in ("bin", "multi")
                    m = co.analysis_mask()
                elif is_arm:
                    from paths import COHORT as _CO
                    if kindname not in _CO[n].get("arm_contrasts", {}):
                        print(f"  {n}/{tname}: no such arm contrast for this cohort, "
                              f"skipped", flush=True)
                        continue
                    y = co.arm_contrast(kindname)
                    # analysis_mask, not labelled_mask: an arm is known for every
                    # randomised participant, so requiring a cognitive outcome too would
                    # discard usable subjects for no reason.
                    m = co.analysis_mask()
                else:
                    m = co.labelled_mask(label)
                    y = (co.labels(label) if binary else
                         co.baseline_levels(label) if kindname == "level" else
                         # Same mask as the T1 arm on purpose: requiring the change score
                         # to exist keeps n identical across the two sessions, which is
                         # what makes the pair an internal replication rather than two
                         # analyses on two samples.
                         co.followup_levels(label) if kindname == "level_t2" else
                         co.change_scores(label))
                m = m & np.isfinite(y)
                if (args.exclude_near_zero > 0 and not is_arm
                        and kindname in ("change", "bi")):
                    ch = np.abs(co.change_scores(label))
                    thr = np.nanquantile(ch[m], args.exclude_near_zero)
                    keep = m & (ch >= thr)
                    print(f"    {n}/{tname}: --exclude_near_zero "
                          f"{args.exclude_near_zero:g} drops {int(m.sum() - keep.sum())} "
                          f"of {int(m.sum())} (|change| < {thr:.3f})", flush=True)
                    m = keep
                if m.sum() < 40:
                    continue
                F = feats[n][m]
                yv = y[m]
                if binary and len(set(yv.astype(int))) < 2:
                    continue
                s, sd, oofs = cv_score(F, yv, binary, n_pca=args.n_pca)
                rec = {"n": int(m.sum()), "score": s, "score_sd": sd}
                if binary:
                    rec.update({("model_" + k): v for k, v in
                                compute_metrics(yv, oofs[0]).items()})
                    rec.update({("majority_" + k): v for k, v in
                                majority_baseline(yv, yv).items()})
                else:
                    rec["pearson"] = float(stats.pearsonr(yv, oofs[0]).statistic)
                if args.n_perm:
                    p, mu, sg = permutation_p(F, yv, binary, s, args.n_perm,
                                              n_pca=args.n_pca)
                    rec.update({"p_perm": p, "null_mean": mu, "null_sd": sg})
                per[n] = rec
                # z-score the target within cohort so cohorts on different scales pool
                # Pool the RAW feature, not the per-cohort PCA scores. Each cohort's PCA
                # has its own basis (and its own width, min(n_pca, n-1) = 94/73/100 here),
                # so stacking the scores would put unrelated directions in the same column.
                # Same failure mode as fitting a co-skewness basis per scan.
                pooled_F.append(feats[n][m])
                pooled_y.append(yv if binary else (yv - yv.mean()) / max(yv.std(), 1e-12))
                pooled_c += [n] * int(m.sum())
                flag = ""
                if binary:
                    flag = (f"  ACC {rec['model_acc']:.3f} vs majority "
                            f"{rec['majority_acc']:.3f}")
                print(f"  {kind:8s} {tname:11s} {n:8s} n={rec['n']:3d} "
                      f"{'AUC' if binary else 'rho'} {s:+.4f} +/-{sd:.4f}"
                      + (f"  perm p {rec['p_perm']:.3f}" if args.n_perm else "") + flag,
                      flush=True)
            # --- pooled fit across cohorts ------------------------------------------
            if args.pooled and len(pooled_y) > 1:
                yp = np.concatenate(pooled_y)
                cp = np.array(pooled_c)
                dm = pd.get_dummies(cp, drop_first=True).astype(float).values
                # Raw features first, dummies last: fold_transform reduces only the
                # feature block and appends the dummies untouched.
                Fp = np.hstack([np.vstack(pooled_F), dm])
                n_pass = dm.shape[1]
                sp, sdp, oofp = cv_score(Fp, yp, binary, n_pca=args.n_pca, n_pass=n_pass)
                recp = {"n": int(len(yp)), "score": sp, "score_sd": sdp,
                        "n_cohorts": int(len(set(cp)))}
                if args.n_perm:
                    pv, mu, sg = permutation_p(Fp, yp, binary, sp, args.n_perm,
                                               n_pca=args.n_pca, n_pass=n_pass)
                    recp.update({"p_perm": pv, "null_mean": mu, "null_sd": sg})
                rows.append({"kind": kind, "target": tname, "cohort": "POOLED",
                             "readout": args.readout if kind == "brainlm" else None,
                             "layer": args.layer if kind == "brainlm" else None,
                             "input": args.input, **recp})
                print(f"  {kind:8s} {tname:11s} {'POOLED':8s} n={len(yp):3d} "
                      f"{'AUC' if binary else 'rho'} {sp:+.4f} +/-{sdp:.4f}"
                      + (f"  perm p {recp['p_perm']:.3f}" if args.n_perm else ""),
                      flush=True)
            for n, rec in per.items():
                rows.append({"kind": kind, "target": tname, "cohort": n,
                             "readout": args.readout if kind == "brainlm" else None,
                             "layer": args.layer if kind == "brainlm" else None,
                             "input": args.input, **rec})

            # --- cross-cohort transfer: fit on one, score the others -------------------
            # Binary targets were excluded here, which made --cross_cohort a silent no-op
            # for exactly the target Reviewer 2's M2 is about ("train on one intervention
            # dataset and test on the other"). Both branches now run; the binary one fits a
            # logistic model and reports AUC on the held-out cohort.
            if args.cross_cohort and len(per) > 1:
                def _y(c, mask_only=False):
                    if is_demo:
                        # Demographics live in per-cohort covariate tables, not in the
                        # outcome CSV, so they cannot go through Cohort.labels -- `label`
                        # is the literal string "demo" here and that raised KeyError,
                        # which is what made --cross_cohort crash for every demo target.
                        from rct_positive_control import CONTROLS
                        yy, mm = CONTROLS[c](coh[c].subjects).get(kindname, (None, None))
                        if yy is None:
                            # Not every control exists in every cohort -- site is ACT only,
                            # cortical thickness is CogTE only. An all-NaN vector makes the
                            # finite-mask below drop the cohort instead of crashing.
                            return np.full(len(coh[c]), np.nan)
                        yy = np.asarray(yy, dtype=float)
                        if mm == "multi":
                            vals, cnt = np.unique(yy[np.isfinite(yy)], return_counts=True)
                            yy = np.where(np.isfinite(yy),
                                          (yy == vals[cnt.argmax()]).astype(float), np.nan)
                        return yy
                    if is_arm:
                        return coh[c].arm_contrast(kindname)
                    if binary:
                        return coh[c].labels(label)
                    return (coh[c].baseline_levels(label) if kindname == "level"
                            else coh[c].change_scores(label))
                for src in per:
                    ms = (coh[src].analysis_mask() if is_demo
                          else coh[src].labelled_mask(label))
                    ys = _y(src)
                    ms = ms & np.isfinite(ys)
                    if binary and len(set(ys[ms].astype(int))) < 2:
                        continue
                    Fs = feats[src][ms]
                    # Standardise then reduce (see fold_transform); the source cohort's
                    # scaler and basis are what the target cohort gets projected onto.
                    sc = StandardScaler().fit(Fs)
                    pc = PCA(n_components=min(args.n_pca, int(ms.sum()) - 1),
                             random_state=0).fit(sc.transform(Fs))
                    Zs = pc.transform(sc.transform(Fs))
                    if binary:
                        mdl = LogisticRegression(C=0.01, max_iter=5000).fit(
                            Zs, ys[ms].astype(int))
                    else:
                        yz = (ys[ms] - ys[ms].mean()) / max(ys[ms].std(), 1e-12)
                        mdl = Ridge(alpha=1.0).fit(Zs, yz)
                    for dst in per:
                        if dst == src:
                            continue
                        md = (coh[dst].analysis_mask() if is_demo
                              else coh[dst].labelled_mask(label))
                        yd = _y(dst)
                        md = md & np.isfinite(yd)
                        Z = pc.transform(sc.transform(feats[dst][md]))
                        if binary:
                            pr = mdl.predict_proba(Z)[:, 1]
                            met = _metric(yd[md].astype(int), pr, True)
                            mname = "auc"
                        else:
                            pr = mdl.predict(Z)
                            met = float(stats.spearmanr(yd[md], pr).statistic)
                            mname = "spearman"
                        # A null is not optional here. The model is already fitted, so
                        # shuffling the TEST cohort's labels is nearly free, and without it
                        # a single transfer cell at AUC 0.71 on n=74 reads as a finding when
                        # its confidence interval covers chance. 24 cells are computed, so
                        # the per-cell p is also Holm-corrected downstream.
                        rr = np.random.default_rng(0)
                        yt = (yd[md].astype(int) if binary else yd[md])
                        nul = np.empty(1000)
                        for i in range(1000):
                            yp = rr.permutation(yt)
                            nul[i] = (_metric(yp, pr, True) if binary
                                      else float(stats.spearmanr(yp, pr).statistic))
                        pv = float((nul >= met).mean())
                        cross_rows.append({"kind": kind, "target": tname, "train": src,
                                           "test": dst, "n_test": int(md.sum()),
                                           mname: met, "p_perm": pv,
                                           "null_mean": float(np.nanmean(nul))})
                        print(f"    cross {kind:8s} {tname:11s} {src:>7s} -> "
                              f"{dst:<8s} n={int(md.sum()):3d} "
                              f"{'AUC' if binary else 'rho'} {met:+.4f}  perm p {pv:.3f}",
                              flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"protocol_version": PROTOCOL_VERSION, "args": vars(args), "rows": rows, "cross_cohort": cross_rows},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")

    if rows:
        df = pd.DataFrame(rows)
        print("\nbest kind per target (mean over cohorts):")
        g = df.groupby(["target", "kind"])["score"].mean().reset_index()
        for t in df["target"].unique():
            sub = g[g["target"] == t].sort_values("score", ascending=False)
            line = "  ".join(f"{r.kind} {r.score:+.4f}" for r in sub.itertuples())
            print(f"  {t:12s} {line}")


if __name__ == "__main__":
    main()
