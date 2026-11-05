#!/usr/bin/env python3
"""Fit the dataset-level constant BrainLM's pretraining input subtracts (step 2).

`src/brainlm_input.py` documents what the pretraining distribution actually is, read off
the code that produced it. Two things stand between our data and that distribution:

  step 1  per-ROI amplitude. Our tensors are z-scored within scan, which divided it out.
          It cannot be fitted -- it has to be supplied, as the per-ROI standard deviations
          used in the z-scoring (`AmplitudeStats`). This script cannot manufacture it.
  step 2  a per-ROI constant, median / IQR over the whole dataset. That *can* be fitted,
          and this script fits it.

Fitting step 2 on data that has not had step 1 restored reproduces the operation but not
the pretraining numbers, so the output records `source` and every consumer prints it.

Examples:
  python scripts/fit_input_stats.py --scan_dir data/adni/raw
  python scripts/fit_input_stats.py --cohort act
  python scripts/fit_input_stats.py --cohort act --sd_from_csv roi_sd.csv   # supply step 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from brainlm_input import AmplitudeStats, fit_voxel_offset  # noqa: E402
from cohorts import load_cohort  # noqa: E402
from paths import ADNI_SCAN_DIR, COHORT, DERIVED_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Fit BrainLM input statistics")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scan_dir", nargs="?", const=str(ADNI_SCAN_DIR),
                   help=f"directory of [R,T] .pt scans (default {ADNI_SCAN_DIR})")
    g.add_argument("--cohort", choices=sorted(COHORT), help="an RCT cohort tensor")
    ap.add_argument("--sd_from_csv", default=None,
                    help="CSV of per-ROI standard deviations used in the original "
                         "z-scoring; supplying it restores step 1 before step 2 is fitted")
    ap.add_argument("--limit", type=int, default=None, help="use at most this many scans")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.scan_dir:
        files = sorted(Path(args.scan_dir).glob("*.pt"))[: args.limit]
        if not files:
            raise SystemExit(f"no .pt files under {args.scan_dir}")
        scans = [torch.load(f, map_location="cpu", weights_only=False).float() for f in files]
        tag = Path(args.scan_dir).name
        print(f"{len(scans)} scans from {args.scan_dir}", flush=True)
    else:
        coh = load_cohort(args.cohort, require_labels=False)
        x = coh.bold.float()
        # Usable frames only: the NaN padding would poison every quantile.
        scans = [x[i, :, : int(coh.valid[i, tp]), tp]
                 for tp in (0, 1) for i in range(x.shape[0])][: args.limit]
        tag = args.cohort
        print(f"{len(scans)} scans (both timepoints) from cohort {args.cohort}", flush=True)

    # --- step 1, if the caller can supply it -------------------------------------------
    if args.sd_from_csv:
        sd = torch.as_tensor(np.loadtxt(args.sd_from_csv, delimiter=",", dtype=np.float32))
        if sd.ndim != 1 or sd.shape[0] != scans[0].shape[0]:
            raise SystemExit(f"{args.sd_from_csv} must hold {scans[0].shape[0]} values, got {tuple(sd.shape)}")
        amp = AmplitudeStats(sd, "measured")
        scans = [amp.apply(s) for s in scans]
        amp_out = Path(args.out).with_name(f"{tag}_amplitude.pt") if args.out else \
            DERIVED_DIR / "stats" / f"{tag}_amplitude.pt"
        amp_out.parent.mkdir(parents=True, exist_ok=True)
        amp.save(amp_out)
        print(f"step 1 restored from {args.sd_from_csv}; wrote {amp_out}")
        source = "amplitude_restored"
    else:
        source = "amplitude_missing"
        print("step 1 NOT restored: no --sd_from_csv given, so the input is still z-scored\n"
              "  and off the pretraining distribution. See src/brainlm_input.py.")

    # --- step 2 -------------------------------------------------------------------------
    off = fit_voxel_offset(scans, source=source)
    out = Path(args.out) if args.out else DERIVED_DIR / "stats" / f"{tag}_offset.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    off.save(out)
    print(f"\nsource={off.source}")
    print(f"  offset (median/IQR) per ROI: mean {off.offset.mean():+.4f}  "
          f"range [{off.offset.min():+.4f}, {off.offset.max():+.4f}]")
    if source == "amplitude_missing":
        print("\nNOTE: fitted on z-scored tensors. This reproduces the operation, not the\n"
              "pretraining numbers. Results using it are an approximation and must say so.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
