#!/usr/bin/env python3
"""Does the FM embedding encode a cardiovascular trait? BREATHE's out-of-scanner HRV.

WHY THIS TARGET AND NOT THE OTHER TWO. Three HRV files exist for BREATHE. Reviewing them
settled which one can carry a result:

  BREATHE_MRI_HRV_windows_101s_step20.2s.csv   in-scanner, per-window. Mean HR is plausible
      in 99.2% of windows; RMSSD is plausible in 1.6% (median 213 ms against a 15-60 ms
      normal range), and 29% even in the file's own cleanest subset. Peak COUNTING survives
      the scanner, peak TIMING does not -- the gradient-artifact signature. Worse, the
      101 s window stepped by 20.2 s is 80% overlapping, so its 14-point series has lag-1
      +0.812 (guaranteed by the shared data) and lag-5 -0.403 at the first NON-overlapping
      pair. Sliding a 101 s window over a ~300 s record leaves ~3 independent chunks, so the
      series is an over-smoothed function of ~3 numbers, not 14 observations. And requiring
      a real alignment marker leaves 23 subjects.
  BREATHE_MRI_RMSSD.csv                        in-scanner summary. 6.0% of runs pass its own
      Usable flag; RMSSD median 214 ms.
  BREATHE_Assessment_RMSSD.csv                 out-of-scanner. 93% Usable, ArtifactPct
      median 0.56%, RMSSD median 28.6 ms, RMSSD/MeanNN 0.0405 -- all physiologically normal.
      104 imaged subjects, 100 at T1 and 93 at T2.

So the in-scanner question -- "does the representation encode scan-time physiology" --
cannot be asked with these files, and this script asks the answerable one instead: **does the
representation encode a cardiovascular TRAIT**, measured on a different day outside the
scanner. That is a weaker claim and it is stated as such.

WHY MeanNN IS PRIMARY AND RMSSD IS NOT. Three independent lines agree:

  in-scanner plausibility   MeanHR 99.2% vs RMSSD 1.6%
  measurement theory        a mean of ~28 beats has ~3% relative error; a second-order
                            statistic of their successive differences has ~14%, and
                            timing jitter adds in quadrature to the differences
  T1-T2 retest (n=89)       MeanNN +0.776, MedianNN +0.771, LFHF +0.534, RMSSD +0.388,
                            SDNN +0.381, HF +0.326, LF +0.296

The last one is the ceiling in §5b's sense: a representation cannot predict a target better
than the target predicts itself across sessions. RMSSD's own retest is 0.388, so it is
reported WITH that number attached and cannot support a strong null either way.

DESIGN. Every arm is `demo` plus ONE feature block reduced to the same k, so
FM-versus-amp and FM-versus-FC are dimension-matched BY CONSTRUCTION rather than needing
§1b's 2k control. `amp` is the comparison that matters: cardiac pulsatility modulates
per-ROI BOLD amplitude directly, and `amp` is a 424-dim closed-form read of exactly that.
One fusion pair is included with its 2k control. T1 and T2 are analysed separately, because
§5d showed that a probe appearing at one session and not the other is a session effect.
Demographics are carried as covariates rather than residualised out; age explains little of
this target anyway (RMSSD vs age +0.185 at T1, +0.181 at T2, sex ~0.00).
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

#: T1-T2 Spearman of the target with itself, n=89. The §5b ceiling, printed beside
#: every score so a null is never quoted against an unstated bound.
RETEST = {"HRV_MeanNN": 0.776, "HRV_MedianNN": 0.771, "HRV_LFHF": 0.534,
          "HRV_RMSSD": 0.388, "HRV_SDNN": 0.381, "HRV_HF": 0.326, "HRV_LF": 0.296}


def subj_key(x):
    """BREATHE imaging IDs are `sub-101`, the HRV files use `s101`."""
    m = re.search(r"(\d+)", str(x))
    return m.group(1).lstrip("0") if m else None


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hrv", default="data/rct/breathe/BREATHE_Assessment_RMSSD.csv")
    ap.add_argument("--targets", nargs="+", default=["HRV_MeanNN", "HRV_RMSSD"],
                    help="primary first; RMSSD is secondary and ceiling-limited")
    ap.add_argument("--timepoints", nargs="+", default=["T1", "T2"])
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--k", type=int, default=10, help="components per feature block")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--out", default="runs/v2/rct_hrv_trait.json")
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

    hrv = pd.read_csv(a.hrv)
    hrv["k"] = hrv["Subject"].map(subj_key)
    hrv = hrv[(hrv.Condition == "rest") & (hrv.Usable)]

    #: Features per timepoint. T1 HRV is paired with the T1 scan, T2 with the T2 scan --
    #: the closest to concurrent these files allow, though the assessment is on a separate
    #: day, which is why this measures a trait rather than scan-time physiology.
    print("building features", flush=True)
    F = {}
    gpu_acquire()
    model = load_brainlm_forpretraining(Path(a.pretrained)).to(DEVICE).eval()
    for ti, tp in enumerate(a.timepoints):
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

    rows, tests = [], []
    for tgt in a.targets:
        for tp in a.timepoints:
            h = hrv[hrv.Timepoint == tp][["k", tgt]].dropna().drop_duplicates("k")
            m = np.isin(keys, h["k"].values) & np.isfinite(demo).all(1)
            if m.sum() < 30:
                print(f"  {tgt}/{tp}: n={int(m.sum())}, skipped")
                continue
            order = {kk: v for kk, v in zip(h["k"], h[tgt])}
            y = np.array([order[kk] for kk in keys[m]], float)
            d = demo[m]
            ceil = RETEST.get(tgt, float("nan"))
            print(f"\n=== {tgt} / {tp}   n={len(y)}   target's own retest ceiling {ceil:.3f} ===",
                  flush=True)

            k = a.k
            # Bare blocks as well as demo+block. `demo` is nearly uninformative for this
            # target (MeanNN vs age -0.040, vs sex -0.110), so carrying it in every arm
            # adds two noise columns and depresses every absolute score. Without the bare
            # arms a null could not be distinguished from the covariates masking a signal.
            arms = {"demo": ([d], [2])}
            for fam in ("FM", "amp", "FC", "tucker"):
                arms[f"{fam}({k})"] = ([F[(tp, fam)][m]], [k])
            for fam in ("FM", "amp", "FC", "tucker"):
                arms[f"demo+{fam}({k})"] = ([d, F[(tp, fam)][m]], [2, k])
            arms[f"demo+FC({2 * k})"] = ([d, F[(tp, "FC")][m]], [2, 2 * k])
            arms[f"demo+FC({k})+FM({k})"] = ([d, F[(tp, "FC")][m], F[(tp, "FM")][m]],
                                             [2, k, k])
            oof = {}
            for nm, (blocks, ks) in arms.items():
                p = np.mean([oof_predictions(blocks, ks, y, s) for s in a.seeds], axis=0)
                oof[nm] = p
                sc = score(y, p)
                rows.append(dict(target=tgt, timepoint=tp, arm=nm, n=int(len(y)),
                                 score=sc, ceiling=ceil))
                print(f"  {nm:24s} rho {sc:+.4f}   ({100 * sc / ceil:>5.1f}% of ceiling)"
                      if ceil == ceil else f"  {nm:24s} rho {sc:+.4f}", flush=True)

            PAIRS = [("demo", f"demo+FM({k})", "does the FM add over age+sex?"),
                     (f"demo+amp({k})", f"demo+FM({k})",
                      "THE key test: amp is a closed-form read of what cardiac pulsatility modulates"),
                     (f"demo+FC({k})", f"demo+FM({k})", "FM vs FC, matched k"),
                     (f"demo+tucker({k})", f"demo+FM({k})", "FM vs tucker-FC, matched k"),
                     ("demo", f"demo+amp({k})", "does amp add over age+sex?"),
                     (f"demo+FC({2 * k})", f"demo+FC({k})+FM({k})",
                      "fusion, dimension-matched control"),
                     (f"amp({k})", f"FM({k})", "the key test again, without the demo block"),
                     (f"FC({k})", f"FM({k})", "FM vs FC, no demo block")]
            for u, v, note in PAIRS:
                if u not in oof or v not in oof:
                    continue
                dd, p, ci = paired_bootstrap(y, oof[u], oof[v], a.n_boot)
                tests.append(dict(target=tgt, timepoint=tp, a=u, b=v, note=note,
                                  delta=dd, p=p, ci_lo=ci[0], ci_hi=ci[1], n=int(len(y))))
                print(f"  {v:24s} vs {u:20s} delta {dd:+.4f} "
                      f"[{ci[0]:+.4f},{ci[1]:+.4f}] p {p:.4f}{' *' if p < 0.05 else '  '}",
                      flush=True)
                print(f"      {note}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"args": vars(a), "retest_ceiling": RETEST,
                                       "scores": rows, "tests": tests}, indent=1))
    print("\n" + "=" * 88)
    print("REPLICATION ACROSS SESSIONS (§5d discipline: a cell at one session only is a")
    print("session effect, not a representation reading physiology)")
    print("=" * 88)
    for tgt in a.targets:
        ks = list(dict.fromkeys((t["a"], t["b"]) for t in tests if t["target"] == tgt))
        for u, v in ks:
            cs = {t["timepoint"]: t for t in tests
                  if t["target"] == tgt and t["a"] == u and t["b"] == v}
            if len(cs) < 2:
                continue
            s = "  ".join(f"{tp} {cs[tp]['delta']:+.4f} (p {cs[tp]['p']:.3f})"
                          for tp in a.timepoints if tp in cs)
            both = all(cs[tp]["delta"] > 0 for tp in cs)
            print(f"  {tgt:12s} {v:22s} vs {u:20s} {s}"
                  f"   {'both positive' if both else 'NOT consistent'}")
    print(f"\nwrote {a.out}")
    print("\nEvery score above must be read against the target's own T1-T2 retest, printed")
    print("with it. Exceeding that bound would indicate a leak, not a result.")


if __name__ == "__main__":
    main()
