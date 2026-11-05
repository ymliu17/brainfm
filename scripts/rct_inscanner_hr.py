#!/usr/bin/env python3
"""Does the FM embedding encode SCAN-TIME heart rate? The concurrent physiological question.

WHY THIS IS THE RIGHT TARGET, after reviewing four HRV files for BREATHE.

Beat-to-beat variability is destroyed in MOST windows, but -- corrected 2026-09-17 -- not in
all of them, and the distinction was lost in the original version of this paragraph.

Over ALL windows the variability metrics are indeed hopeless: `MedianASD` (the MEDIAN
absolute successive difference, immune to a minority of bad beats) is 161.6 ms against a
~20-30 ms norm, RMSSD 209.1 against 15-60, pNN50 66.7% against 5-25%. From that I concluded
that "the contamination is PERVASIVE rather than outlier-driven, so no robust variability
estimator can rescue it". THAT INFERENCE WAS WRONG, because it was drawn over rows the
quality flag already rejects. Within `Usable_HR` (539 of 4844 BREATHE windows, 11.1%) the
same medians are MedianASD 28.9 ms and pNN50 17.4%, both inside the normal range; only RMSSD
stays high at 71.7, which is the expected signature since RMSSD is the non-robust one of the
three. pSOPT is cleaner again: 56.8% of windows pass, and within them MedianASD is 17.8 ms,
RMSSD 28.7 and pNN50 7.7%, all normal.

So the correct statement is that contamination is pervasive ACROSS windows -- most windows
fail -- while WITHIN the passing windows beat timing is clean. Variability targets are
therefore not ruled out on the usable subset, as the earlier text claimed.

Mean heart rate remains the best-determined choice and is what this script uses: on
`Usable_HR` windows 100% are physiologically plausible, `Coverage` ~0.99, and `MedianHR`
agrees with `BeatRate` in 305 of 347 runs.

WITHDRAWN 2026-09-17: this docstring previously claimed "the per-window design is also
closed, and by the ECG alone", from a within-scan reliability of +0.212 and a lag-1
autocorrelation of +0.047. Both halves of that argument were wrong and the per-window design
is OPEN. See scripts/rct_inscanner_patch_hr.py, which supersedes it.

  1. The +0.212 was computed over ALL rows of the window file. `Usable_HR` is the flag that
     separates real windows from artefactual ones, and applying it cuts the measurement SE
     from 3.62 to 1.18 bpm while leaving the real spread at ~5.0, i.e. reliability +0.944
     rather than +0.212. The formula `1 - E[SE^2]/E[SD^2]` is actively fooled by the rows
     that flag excludes: a window whose beats are consistently MIScounted has a small SE and
     a large deviation, which the formula reads as signal.
  2. Low autocorrelation is not low reliability. Reliability +0.94 with lag-1 ~ 0 means the
     windows are well-measured and near-INDEPENDENT, which makes the target BETTER -- the
     effective n is ~10 per scan rather than 1. The inference "temporally uncorrelated
     therefore artifact" does not hold once the measurement error is bounded separately.

The scan-level analysis below is unaffected: it only ever used the mean over windows, and a
mean is what both readings agree is well determined.

What survives is the SCAN-level mean, and it is excellent: 106 imaged subjects over 187
scans, median 79.1 bpm, 100% in the plausible 45-110 range, between-scan SD 7.13 bpm and a
standard error on the scan mean of ~1.04 bpm -- internal reliability 0.979.

TWO CEILINGS, AND WHY BOTH MUST BE QUOTED
-----------------------------------------
    0.979   the measurement ceiling: how well the target is determined at all
    0.509   sqrt(test-retest) = sqrt(0.259): the ceiling for a predictor that captures only
            STABLE subject properties

The second is a hard constraint here, not a caveat. In-scanner heart rate is mostly state --
its own T1-T2 retest is 0.259 -- while §4a measured this very representation identifying
BREATHE participants across their own two scans at 0.949 top1, i.e. it is almost perfectly
STABLE within subject. A stable feature cannot correlate with a mostly-unstable target
beyond sqrt(its trait share). **So exceeding ~0.51 between subjects would itself be the
evidence that the representation tracks scan-specific state rather than anatomy.**

ARM B IS THE SHARP TEST. The change score dHR = HR(T2) - HR(T1) removes the subject entirely:
SD 8.68 bpm, SE 1.47 bpm, **reliability 0.971**, n = 81 paired subjects. It is purely state,
it is highly reliable, and it is immune to every between-subject confound §4 identifies --
head size, registration residual and vascular anatomy are constant within a subject. This is
the best-conditioned physiological target in this project.

`amp` IS THE COMPARISON THAT MATTERS, and it is dimension-matched by construction because
every arm carries one feature block reduced to the same k. Cardiac pulsatility modulates
per-ROI BOLD amplitude directly, and `amp` is a 424-dim closed-form read of exactly that. If
the FM beats it, the representation holds cardiac information no closed-form amplitude
statistic captures; if it ties, the information was in the input.

T1 and T2 are analysed separately in Arm A rather than pooled, so no subject appears in both
train and test, and so §5d's replication discipline applies: a cell at one session only is a
session effect.
"""
from __future__ import annotations

import argparse
import json
import re
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
from fusion_increment import oof_predictions, paired_bootstrap, score  # noqa: E402
from mci_axis_transfer import adni_features  # noqa: E402
from paths import OFFICIAL_BRAINLM_13M, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#: Ceilings derived in the docstring. Printed beside every score: a null against an
#: unstated bound is not interpretable (the §5b/§9.2 rule).
CEIL_MEASURE = 0.979          # internal reliability of the scan mean
CEIL_STABLE = 0.509           # sqrt(T1-T2 retest 0.259): bound for a stable predictor
CEIL_DELTA = 0.971            # reliability of the change score


def subj_key(x):
    m = re.search(r"(\d+)", str(x))
    return m.group(1).lstrip("0") if m else None


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hr", default="data/rct/breathe/BREATHE_MRI_MeanHR_windows_20.2s.csv")
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--out", default="runs/v2/rct_inscanner_hr.json")
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen

    import pandas as pd
    from model_loader import load_brainlm_forpretraining
    from rct_positive_control import CONTROLS
    from rct_probe import upsample_short
    from thirdorder import shared_coskew_basis

    co = center_rois(load_cohort("breathe", release="amp", require_labels=False))
    if int(co.valid.min()) < a.tlen:
        co, _ = upsample_short(co, a.tlen, C.TIMEPOINT_PATCHING_SIZE, fill="pad")
    xyz = load_xyz(XYZ_PATH)
    keys = np.array([subj_key(s) for s in co.subjects])
    ctrl = CONTROLS["breathe"](co.subjects)
    demo = np.column_stack([np.asarray(ctrl["age"][0], float),
                            np.asarray(ctrl["sex"][0], float)])

    # Scan-level mean HR from the per-window file: the mean over full 20.2 s windows of the
    # resting run. `ScanMeanHR` in the file is used where present; recomputing from the
    # windows gives the same quantity and makes the window filter explicit.
    w = pd.read_csv(a.hr)
    w["k"] = w["Subject"].map(subj_key)
    w = w[(w.Run == "rest-run01") & (w.SpanSec.round(1) == 20.2)]
    hr = (w.groupby(["k", "Timepoint"])
            .agg(HR=("MeanHR", "mean"), nwin=("MeanHR", "size")).reset_index())
    hr = hr[hr.nwin >= 5]
    print(f"scan-level HR: {len(hr)} scans over {hr.k.nunique()} subjects "
          f"(>=5 full windows)", flush=True)

    print("building features", flush=True)
    F = {}
    gpu_acquire()
    model = load_brainlm_forpretraining(Path(a.pretrained)).to(DEVICE).eval()
    for ti, tp in enumerate(("T1", "T2")):
        x = co.bold[..., ti][:, :, :a.tlen]
        F[(tp, "FM")] = adni_features("brainlm", x, model, xyz, "emb_fc", 3, 64, None)
        F[(tp, "amp")] = adni_features("amp", x, None, xyz, None, None, 64, None)
        F[(tp, "FC")] = adni_features("fc", x, None, xyz, None, None, 64, None)
        b = shared_coskew_basis(x.numpy(), None, 64)
        F[(tp, "tucker")] = adni_features("tucker", x, None, xyz, None, None, 64, b)
        print(f"  {tp} done", flush=True)
    del model
    torch.cuda.empty_cache()
    from benchmark_gpu_phase import release_gpu_for_cpu_tail
    release_gpu_for_cpu_tail()

    FAMS = ("FM", "amp", "FC", "tucker")
    k = a.k
    rows, tests = [], []

    def run_cell(label, y, blocks_by_arm, ceilings):
        oof = {}
        print(f"\n=== {label}   n={len(y)} ===")
        for c_nm, c_v in ceilings.items():
            print(f"  ceiling [{c_nm}] {c_v:.3f}")
        for nm, (blocks, ks) in blocks_by_arm.items():
            p = np.mean([oof_predictions(blocks, ks, y, s) for s in a.seeds], axis=0)
            oof[nm] = p
            sc = score(y, p)
            rows.append(dict(cell=label, arm=nm, n=int(len(y)), score=sc, **ceilings))
            frac = "  ".join(f"{100 * sc / v:>6.1f}% of {n}" for n, v in ceilings.items())
            print(f"  {nm:22s} rho {sc:+.4f}   {frac}", flush=True)
        PAIRS = [(f"amp({k})", f"FM({k})",
                  "THE key test: amp is a closed-form read of what cardiac pulsatility modulates"),
                 (f"FC({k})", f"FM({k})", "FM vs FC, matched k"),
                 (f"tucker({k})", f"FM({k})", "FM vs tucker-FC, matched k")]
        for u, v, note in PAIRS:
            if u not in oof or v not in oof:
                continue
            d, p, ci = paired_bootstrap(y, oof[u], oof[v], a.n_boot)
            tests.append(dict(cell=label, a=u, b=v, note=note, delta=d, p=p,
                              ci_lo=ci[0], ci_hi=ci[1], n=int(len(y))))
            print(f"  {v:22s} vs {u:18s} delta {d:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]"
                  f" p {p:.4f}{' *' if p < 0.05 else ''}", flush=True)
            print(f"      {note}", flush=True)

    # --- ARM A: between subjects, each session separately ----------------------------
    for tp in ("T1", "T2"):
        h = hr[hr.Timepoint == tp].set_index("k")["HR"]
        m = np.isin(keys, h.index.values) & np.isfinite(demo).all(1)
        if m.sum() < 30:
            continue
        y = np.array([h[kk] for kk in keys[m]], float)
        arms = {"demo": ([demo[m]], [2])}
        for f in FAMS:
            arms[f"{f}({k})"] = ([F[(tp, f)][m]], [k])
            arms[f"demo+{f}({k})"] = ([demo[m], F[(tp, f)][m]], [2, k])
        run_cell(f"ARM A between-subject / {tp}", y, arms,
                 {"measure": CEIL_MEASURE, "stable-feature": CEIL_STABLE})

    # --- ARM B: within subject, the change score --------------------------------------
    p1 = hr[hr.Timepoint == "T1"].set_index("k")["HR"]
    p2 = hr[hr.Timepoint == "T2"].set_index("k")["HR"]
    both = [kk for kk in keys if kk in p1.index and kk in p2.index]
    m = np.isin(keys, both)
    if m.sum() >= 30:
        y = np.array([p2[kk] - p1[kk] for kk in keys[m]], float)
        # No demo block: age and sex do not change between the two scans, so they carry no
        # information about a change score and would only add noise columns.
        arms = {f"d{f}({k})": ([F[("T2", f)][m] - F[("T1", f)][m]], [k]) for f in FAMS}
        run_cell("ARM B within-subject dHR", y, arms, {"measure": CEIL_DELTA})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"args": vars(a), "ceilings": {"measure": CEIL_MEASURE, "stable": CEIL_STABLE,
                                       "delta": CEIL_DELTA},
         "scores": rows, "tests": tests}, indent=1))
    print("\n" + "=" * 90)
    print("HOW TO READ THIS")
    print("=" * 90)
    print(f"  ARM A above {CEIL_STABLE:.2f} would mean the representation tracks scan-specific")
    print("  STATE, not just stable anatomy -- §4a measured this same representation")
    print("  identifying these participants across their two scans at 0.949, so a stable")
    print("  feature is bounded by sqrt(the target's trait share).")
    print("  ARM B is the direct state test and carries no such bound: its ceiling is the")
    print(f"  change score's own reliability, {CEIL_DELTA:.3f}.")
    print("  In both arms the FM earns credit only by beating `amp` at matched dimension.")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
