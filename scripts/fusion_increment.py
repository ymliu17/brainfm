#!/usr/bin/env python3
"""Does the foundation model add anything on top of FC? A dimension-matched increment test.

WHY THIS DESIGN
---------------
Two failure modes make a naive fusion test meaningless, and both were hit earlier in this
project:

1. Concatenating raw blocks lets the wide one dominate. FC is 89676-dimensional and the
   amplitude profile is 424, and a plain hstack reproduced FC's score to 4 decimals on
   every target -- the narrow block was ignored entirely. So each block is reduced to the
   SAME number of components first, by a PCA fitted inside the training fold.

2. If FC(k) + FM(k) beats FC(k), that could simply be having 2k components instead of k.
   So FC(2k) is included as a dimension-matched control. The FM only earns credit if
   FC(k)+FM(k) beats BOTH FC(k) and FC(2k).

The comparison is paired: all arms share folds and seeds, and the increment is tested by
bootstrapping SUBJECTS on the out-of-fold predictions, so the null accounts for the fact
that the same subjects appear in every arm.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402
from probe_readouts import brainlm_readout_batched  # noqa: E402
from transfer_baseline import amp_feature, fc_feature  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Public imports retain the API used by clinical and physiological probes.
from benchmark_evaluation import (PROTOCOL_VERSION, strata, oof_predictions,
                                  score, paired_bootstrap)


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description="Dimension-matched FM-on-top-of-FC increment")
    ap.add_argument("--tag", default="raw_ADNI_MEM_first_t140_off")
    ap.add_argument("--in_dir", default=str(DERIVED_DIR / "adni"))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--readout", default="emb_fc")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--k", type=int, default=50, help="components per block")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    C.configure_backends()

    d = Path(args.in_dir)
    x = torch.load(d / f"ADNI_x_{args.tag}.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(d / f"ADNI_y_{args.tag}.pt", map_location="cpu", weights_only=False).numpy()
    import pandas as pd
    manifest = pd.read_csv(d / f"ADNI_manifest_{args.tag}.csv")
    if manifest["PTID"].isna().any():
        raise ValueError("Every ADNI row requires a nonmissing PTID")
    groups = manifest["PTID"].astype(str).to_numpy()
    if len(groups) != len(y) or not np.allclose(manifest["label"], y):
        raise ValueError("ADNI manifest is not aligned with the target tensor")
    xyz = load_xyz(XYZ_PATH)
    print(f"{args.tag}: X={tuple(x.shape)}  n={len(y)}  "
          f"{'binary' if len(np.unique(y)) == 2 else 'continuous/ordinal'} target", flush=True)

    gpu_acquire()
    model = load_brainlm_forpretraining(Path(args.pretrained)).to(DEVICE).eval()
    F_fm = brainlm_readout_batched(model, x, xyz, args.layer, args.readout)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    from benchmark_gpu_phase import release_gpu_for_cpu_tail
    release_gpu_for_cpu_tail()
    F_fc = fc_feature(x.numpy(), None)
    F_amp = amp_feature(x.numpy(), None)
    print(f"  blocks: FC {F_fc.shape[1]}  FM({args.readout} L{args.layer}) {F_fm.shape[1]}  "
          f"amp {F_amp.shape[1]}   k={args.k} components each", flush=True)

    k = args.k
    arms = {
        "FC(k)": ([F_fc], [k]),
        "FC(2k)": ([F_fc], [2 * k]),                 # dimension-matched control
        "FM(k)": ([F_fm], [k]),
        "FC(k)+FM(k)": ([F_fc, F_fm], [k, k]),
        "FC(k)+amp(k)": ([F_fc, F_amp], [k, k]),
        "FC(k)+FM(k)+amp(k)": ([F_fc, F_fm, F_amp], [k, k, k]),
        # The cell that decides whether teaching the encoder amplitude could help. If the
        # FM already carries the per-ROI amplitude profile, amp adds nothing on top of it
        # and a new post-training term aimed at amplitude has nothing to add either. FM(2k)
        # is the dimension-matched control, as FC(2k) is for the FC blocks.
        "amp(k)": ([F_amp], [k]),
        "FM(2k)": ([F_fm], [2 * k]),
        "FM(k)+amp(k)": ([F_fm, F_amp], [k, k]),
    }
    preds, scores = {}, {}
    for name, (blks, ks) in arms.items():
        ps = [oof_predictions(blks, ks, y, s, groups=groups) for s in args.seeds]
        preds[name] = np.mean(ps, axis=0)          # seed-averaged out-of-fold prediction
        scores[name] = [score(y, p) for p in ps]
        print(f"  {name:20s} {np.mean(scores[name]):+.4f} +/- {np.std(scores[name]):.4f}",
              flush=True)

    metric = "AUC" if len(np.unique(y)) == 2 else "rho"
    print(f"\n{'=' * 78}\nincrement over each control, paired bootstrap on subjects ({metric})")
    tests = [("FC(k)", "FC(k)+FM(k)"), ("FC(2k)", "FC(k)+FM(k)"),
             ("FC(k)+amp(k)", "FC(k)+FM(k)+amp(k)"), ("FC(k)", "FC(k)+amp(k)"),
             # Does amplitude add to the FM? A large increment means the encoder does NOT
             # already hold it; a null means it does, and an amplitude-flavoured objective
             # would be teaching the encoder something it has.
             ("FM(k)", "FM(k)+amp(k)"), ("FM(2k)", "FM(k)+amp(k)"),
             # amp against the SAME dimension control the FM is held to. Without this pair
             # the amp column is measured against FC(k) while the FM column is measured
             # against FC(2k), which is the confound this script exists to remove.
             ("FC(2k)", "FC(k)+amp(k)"),
             # The head-to-head the sec 1 claim rests on: the SAME operator (correlate the
             # 424 ROIs, upper triangle, same probe, same output dimension), applied to the
             # encoder's per-ROI summaries versus to the raw time courses.
             ("FM(k)", "FC(k)"), ("FM(2k)", "FC(2k)")]
    rows = []
    for a, b in tests:
        obs, p, ci, detail = paired_bootstrap(
            y, preds[a], preds[b], args.n_boot, groups=groups, return_details=True)
        rows.append({"control": a, "arm": b, "delta": obs, "p_one_sided": p,
                     "ci95": list(ci), **detail})
        flag = " *" if p < 0.05 else ""
        print(f"  {b:20s} vs {a:20s}  delta {obs:+.4f}  "
              f"95% CI [{ci[0]:+.4f},{ci[1]:+.4f}]  p {p:.4f}{flag}")
    print("  (the FM earns credit only if it beats BOTH FC(k) and the FC(2k) "
          "dimension-matched control)")
    print("=" * 78)

    out = Path(args.out) if args.out else (RUNS_DIR / "benchmark_v1" / "fusion" / f"{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"protocol_version": PROTOCOL_VERSION, "args": vars(args),
               "subjects": groups.tolist(), "y": y.tolist(),
               "oof_predictions": {k: v.tolist() for k, v in preds.items()},
               "scores": {kk: [float(np.mean(v)), float(np.std(v))] for kk, v in scores.items()},
               "increments": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
