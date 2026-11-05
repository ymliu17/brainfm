#!/usr/bin/env python3
"""The RCT ceiling §5j asked for: can the representation recognise the same trial
participant in their second scan?

WHY THIS IS THE MISSING NUMBER. §5j established that with baseline cognition, age and sex
already in the model, no imaging feature tested adds detectable predictive information
about follow-up cognition -- and that the null must be quoted with a detectability bound
(median 95% CI half-width 0.047 Spearman). A detectability bound says what effect size
could have been seen. It does not say whether the RCT scans carry subject-specific
information AT ALL. Without that, "the FM adds nothing on these trials" cannot be separated
from "these trials' imaging is unusable", and a reviewer will ask exactly that.

§4 answered the same question on ADNI -- 452 subjects with two sessions, chance top1
0.0022, and the best cells reach 0.31-0.43, i.e. 140-193x chance -- and used the answer to
establish the study's central dissociation: identity is highly recoverable while diagnosis
is not, and adaptation improves the first without moving the second. The trials never got
their own version of that measurement. They have T1 and T2 scans per participant, so the
setup is already there.

WHAT IS AND IS NOT COMPARABLE TO §4
-----------------------------------
`top1` depends strongly on n: at a FIXED representation quality, identifying one of 74
people is easier than one of 452. The trials have n = 74-107 against ADNI's 452, so a
higher RCT top1 than §4's is the expected result whatever the representation does, and
comparing the two directly would be meaningless. Two things are reported instead:

  * a SUBSAMPLE column at n = 74, the smallest cohort, averaged over `--n_sub_reps` draws,
    so the three trials are comparable to EACH OTHER at matched n;
  * `z_self` and `I_diff`, which measure how far the true match sits above the crowd for
    each subject and are far less n-dependent than a rank-1 hit rate.

A matched comparison against ADNI needs ADNI recomputed at n = 74 AND at this window; it is
not attempted here, and the bias direction is stated rather than silently ignored.

THE PADDING CONFOUND, AND WHY IT ONLY THREATENS ONE TRIAL
---------------------------------------------------------
At the official 200-frame input, ACT holds 291-300 frames and BREATHE 295-300, so NEITHER
is padded. CogTE holds 95-118, so all 148 of its sessions are zero-padded under the locked
policy -- up to 53% of the window. Zero-padding is a per-session property, and if a
participant's two sessions shared a length it would hand the matcher a free identity cue.
Measured: T1 length equals T2 length for 28% of ACT, 43% of CogTE and 38% of BREATHE
participants, and the distinct (T1, T2) length pairs number 14-20 over n = 74-107, so
length is a weak cue rather than a strong one. It is not left to that argument: `scan
length` is included as its own two-column feature family, so whatever identity the
acquisition length alone carries is measured and printed next to every other family. Any
CogTE cell that does not clearly exceed it is not interpretable.

Two encoders, which also re-tests §4's adaptation finding on the trials: the official
checkpoint, and the a=1 post-trained encoder of §5i. On ADNI, adaptation raised 10 of 15
non-degenerate fingerprint cells by a mean of +0.051 while CN/AD did not move. If the same
holds here, the dissociation is a property of the data rather than of ADNI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from cohorts import center_rois, load_cohort  # noqa: E402
from datasets import load_xyz  # noqa: E402
from fingerprint_adni import READOUTS, fingerprint  # noqa: E402
from paths import OFFICIAL_BRAINLM_13M, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODERS = [("BrainLM 13M pretrained", str(OFFICIAL_BRAINLM_13M)),
            ("BrainLM 13M ft-MEM a=1", "checkpoints/alpha/brainlm13m_adniMEM_level_a1.pt")]


def subsampled(FA, FB, n_sub, reps, seed=0):
    """Mean top1 over `reps` random subject subsets of size `n_sub`.

    Needed because top1 is a rank-1 hit rate against n-1 distractors, so it is not
    comparable across cohorts of different size. Subsetting the SAME feature matrices
    changes only the distractor count.
    """
    if n_sub >= len(FA):
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        ix = rng.choice(len(FA), n_sub, replace=False)
        vals.append(fingerprint(FA[ix], FB[ix])["top1"])
    return float(np.mean(vals)), len(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohorts", nargs="+", default=["act", "cogte", "breathe"])
    ap.add_argument("--gsr", action="store_true",
                    help="Regress each session's global mean timeseries out of every ROI. "
                         "This is a CONTROL for a release difference, not a preprocessing "
                         "preference. paths.py already records global-signal/per-ROI-sd for "
                         "the A424 releases as 0.152 (act) / 0.248 (cogte) / 0.152 "
                         "(breathe) -- 'the A424 release had global signal suppressed' -- "
                         "while psopt's A424 measures 0.711, i.e. in the range of the other "
                         "cohorts' Schaefer (noGSR) releases rather than their A424 ones. "
                         "psopt's coverage.json says `noGSR: true` explicitly. Since that "
                         "axis is worth 0.047 AUC on ADNI CN/AD by this repo's own "
                         "measurement, psopt's encoder numbers are not comparable to the "
                         "other three until it is equalised. Applying this to all four is "
                         "close to a no-op for the three already suppressed.")
    ap.add_argument("--gain_match", action="store_true",
                    help="After mean-centring, divide each SESSION by one scalar -- its "
                         "median per-ROI sd -- putting every session's median per-ROI sd at "
                         "exactly 1.000. This is not a new normalisation: it is precisely "
                         "what prepare_adni.py --amplitude preserve does, and it removes the "
                         "arbitrary BOLD gain while KEEPING the between-ROI amplitude ratios "
                         "BrainLM was pretrained on. Preferred over --release zscore for "
                         "this comparison, because z-scoring also destroys between-ROI "
                         "amplitude, which section 4b showed carries real signal. Needed "
                         "because the four cohorts are not in the same units: median per-ROI "
                         "sd is 0.733 (act) / 0.679 (cogte) / 0.703 (breathe) / 6.684 "
                         "(psopt), against 0.521 in the ADNI _off build that every result in "
                         "this project was obtained on.")
    ap.add_argument("--release", default="amp", choices=("amp", "zscore", "schaefer", "amp_nogsr", "zscore_nogsr", "schaefer_nogsr"),
                    help="Default `amp` reproduces the published run. `zscore` exists for "
                         "ONE purpose: the four cohorts' A424 releases are NOT in the same "
                         "units. Per-ROI sd is 0.733 (act) / 0.679 (cogte) / 0.703 (breathe) "
                         "but 6.684 for psopt, whose per-ROI offset is also 2917 rather than "
                         "~1 -- psopt was delivered as raw BOLD intensity while the other "
                         "three had already been scaled. center_rois removes the OFFSET but "
                         "the official BrainLM recipe applies no division, so a ~10x "
                         "amplitude difference reaches the encoder. Running `zscore` puts "
                         "every per-ROI sd at exactly 1 and so isolates amplitude: if "
                         "psopt's anomaly on the magnitude readouts (roi_var, roi_norm) "
                         "disappears here while the scale-free emb_fc readout is unchanged, "
                         "the anomaly is units and not TR or biology.")
    ap.add_argument("--tlen", type=int, default=200,
                    help="LOCKED: the FM's official input length. Short scans are "
                         "zero-padded after per-ROI mean-centring, never stretched.")
    ap.add_argument("--layers", default="0,1,2,3,4")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_sub", type=int, default=74,
                    help="matched-n subsample size for cross-cohort comparison")
    ap.add_argument("--n_sub_reps", type=int, default=50)
    ap.add_argument("--out", default="runs/v2/fingerprint_rct.json")
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen
    layers = [int(x) for x in a.layers.split(",")]
    xyz = load_xyz(XYZ_PATH)

    from probe_readouts import extract
    from rct_probe import upsample_short
    from thirdorder import shared_coskew_basis, tucker_fc_feature
    from transfer_baseline import amp_feature, fc_feature

    rows = []
    for cname in a.cohorts:
        co = center_rois(load_cohort(cname, release=a.release, require_labels=False))
        vraw = np.asarray(co.valid).copy()
        if a.gsr:
            # Per session: y_roi <- y_roi - b * gs, with gs the mean over ROIs and b the OLS
            # slope per ROI. Computed on valid frames only so the NaN pad cannot enter it.
            gs_before, gs_after = [], []
            for i in range(co.bold.shape[0]):
                for s in range(co.bold.shape[3]):
                    v = int(vraw[i, s])
                    x = co.bold[i, :, :v, s]
                    ok = torch.isfinite(x).all(1) & (x.std(dim=1) > 1e-6)
                    if ok.sum() < 50:
                        continue
                    gs = x[ok].mean(dim=0)
                    gs = gs - gs.mean()
                    den = (gs * gs).sum().clamp_min(1e-12)
                    xc = x - x.mean(dim=1, keepdim=True)
                    b = (xc @ gs) / den
                    gs_before.append(float(gs.std() / x[ok].std(dim=1).median()))
                    co.bold[i, :, :v, s] = xc - torch.outer(b, gs)
                    y = co.bold[i, :, :v, s][ok]
                    g2 = y.mean(dim=0)
                    gs_after.append(float(g2.std() / y.std(dim=1).median()))
            print(f"  --gsr: global-signal/per-ROI-sd {np.median(gs_before):.3f}"
                  f" -> {np.median(gs_after):.3f}", flush=True)
        if a.gain_match:
            # One scalar per session, computed on that session's VALID frames only so the
            # NaN padding region cannot enter the statistic.
            for i in range(co.bold.shape[0]):
                for s in range(co.bold.shape[3]):
                    v = int(vraw[i, s])
                    sd = co.bold[i, :, :v, s].std(dim=1, unbiased=False)
                    g = float(sd.median().clamp_min(1e-6))
                    co.bold[i, :, :, s] = co.bold[i, :, :, s] / g
            chk = co.bold[0, :, :int(vraw[0, 0]), 0].std(dim=1, unbiased=False).median()
            print(f"  --gain_match: median per-ROI sd now {float(chk):.4f} "
                  f"(session 0, target 1.0000)", flush=True)
        if int(co.valid.min()) < a.tlen:
            co, _ = upsample_short(co, a.tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
        A = co.bold[..., 0][:, :, :a.tlen].contiguous()
        B = co.bold[..., 1][:, :, :a.tlen].contiguous()
        n = len(A)
        pad = int((vraw < a.tlen).sum())
        print(f"\n{'=' * 78}\n{cname}: n={n} participants, chance top1 {1 / n:.4f}"
              f"   sessions padded: {pad}/{2 * n}\n{'=' * 78}", flush=True)

        fam = {}
        an, bn = A.numpy(), B.numpy()
        fam["FC edges"] = (fc_feature(an, None), fc_feature(bn, None))
        fam["amp profile"] = (amp_feature(an, None), amp_feature(bn, None))
        basis = shared_coskew_basis(an, None, rank=64)
        fam["tucker-FC"] = (tucker_fc_feature(an, None, rank=64, basis=basis),
                            tucker_fc_feature(bn, None, rank=64, basis=basis))
        # The confound as its own family. Two columns so `fingerprint`'s correlation across
        # feature dimensions is defined at all; a single column would make every pairwise
        # correlation +-1 and the metric meaningless.
        fam["scan length (CONTROL)"] = (
            np.column_stack([vraw[:, 0], vraw[:, 0] ** 2]).astype(float),
            np.column_stack([vraw[:, 1], vraw[:, 1] ** 2]).astype(float))

        for label, ck in ENCODERS:
            from model_loader import load_brainlm_forpretraining
            m = load_brainlm_forpretraining(Path(ck)).to(DEVICE).eval()
            for li in layers:
                fa = extract(m, A, xyz, [li], list(READOUTS), a.batch_size)
                fb = extract(m, B, xyz, [li], list(READOUTS), a.batch_size)
                for r in READOUTS:
                    fam[f"{label} | {r} L{li}"] = (fa[(li, r)], fb[(li, r)])
                print(f"  {label} layer {li} done", flush=True)
            del m
            torch.cuda.empty_cache()

        print(f"\n  {'family':<40}{'dim':>8}{'top1':>8}{'xchance':>9}"
              f"{'n=74':>8}{'I_diff':>9}{'z_self':>8}")
        for k, (FA, FB) in fam.items():
            FA, FB = np.asarray(FA, float), np.asarray(FB, float)
            d = fingerprint(FA, FB)
            sub, reps = subsampled(FA, FB, a.n_sub, a.n_sub_reps)
            # `fingerprint()` already returns "n", so the cohort size goes in under its
            # own name rather than colliding with it.
            rows.append(dict(cohort=cname, family=k, dim=int(FA.shape[1]), n_subjects=n,
                             chance=1.0 / n, top1_sub=sub, sub_n=a.n_sub,
                             nan_frac=float(np.isnan(FA).mean() + np.isnan(FB).mean()) / 2,
                             **d))
            print(f"  {k:<40}{FA.shape[1]:>8}{d['top1']:>8.3f}{d['top1'] * n:>8.0f}x"
                  f"{(f'{sub:.3f}' if sub == sub else '  -'):>8}"
                  f"{d['I_diff']:>9.4f}{d['z_self']:>8.2f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "rows": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
