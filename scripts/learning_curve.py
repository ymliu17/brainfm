#!/usr/bin/env python3
"""Is ADNI's sample size the binding constraint, or is it the signal?

WHY THIS BEFORE ANYTHING ELSE
-----------------------------
Several remaining ideas amount to "use more of ADNI" -- all sessions instead of one per
subject (1877 vs ~710 scans), more targets, multi-task. They are only worth building if
performance is still rising with n. This subsamples SUBJECTS and traces the curve for both
the foundation model and the trivial baselines, so the answer is measured rather than
assumed.

Reading it:
  * still rising at 100%    -> more data is the lever; the all-sessions and multi-task work
                              has headroom
  * flat from ~50% onward   -> the limit is signal-to-noise in the target or the
                              representation, and adding rows cannot move it. Effort should
                              go to target choice (e.g. the structural volumes, whose
                              visit-to-visit reliability is 0.96-0.99 against 0.92 for the
                              best cognitive composite) rather than to sample size.

Subsampling is by SUBJECT, and the evaluation folds are rebuilt inside each subsample, so
each point is an honest cross-validated score at that n rather than a fixed test set scored
by a smaller training set.

The fMRI timeseries input is untouched: every arm reads the same `official`-recipe build.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from model_loader import load_brainlm_forpretraining  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402
from probe_readouts import _encoder, extract, probe  # noqa: E402
from transfer_baseline import amp_feature, fc_feature  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description="Learning curve over ADNI subjects")
    ap.add_argument("--tag", default="raw_DX_first_t140_off")
    ap.add_argument("--in_dir", default=str(DERIVED_DIR / "adni"))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--readout", default="emb_fc")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.4, 0.55, 0.7, 0.85, 1.0])
    ap.add_argument("--reps", type=int, default=3, help="independent subsamples per fraction")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    C.configure_backends()

    d = Path(args.in_dir)
    x = torch.load(d / f"ADNI_x_{args.tag}.pt", map_location="cpu", weights_only=False).float()
    y = torch.load(d / f"ADNI_y_{args.tag}.pt", map_location="cpu", weights_only=False).numpy()
    man = pd.read_csv(d / f"ADNI_manifest_{args.tag}.csv")
    xyz = load_xyz(XYZ_PATH)
    binary = len(np.unique(y)) == 2
    print(f"{args.tag}: n={len(y)} {'binary (AUC)' if binary else 'graded (rho)'}", flush=True)

    # Features are computed ONCE on everything; subsampling then only selects rows, so the
    # curve isolates sample size and cannot be contaminated by re-extraction differences.
    gpu_acquire()
    model = load_brainlm_forpretraining(Path(args.pretrained)).to(DEVICE).eval()
    n_layers = len(_encoder(model).encoder.layer)
    layer = n_layers if args.layer < 0 else args.layer
    F_fm = extract(model, x, xyz, [layer], [args.readout], args.batch_size)[(layer, args.readout)]
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    from benchmark_gpu_phase import release_gpu_for_cpu_tail
    release_gpu_for_cpu_tail()
    feats = {f"brainlm_{args.readout}_L{layer}": F_fm,
             "fc_edges": fc_feature(x.numpy(), None),
             "amp_profile": amp_feature(x.numpy(), None)}
    print("  features: " + "  ".join(f"{k} {v.shape[1]}" for k, v in feats.items()), flush=True)

    subs = man["PTID"].values if "PTID" in man.columns else np.arange(len(y)).astype(str)
    uniq = np.unique(subs)
    rows = []
    for frac in args.fracs:
        k = max(40, int(round(frac * len(uniq))))
        for rep in range(args.reps):
            rng = np.random.default_rng(1000 * rep + int(frac * 100))
            take = set(rng.permutation(uniq)[:k])
            sel = np.array([s in take for s in subs])
            if binary and len(np.unique(y[sel])) < 2:
                continue
            for name, Fall in feats.items():
                m, s = probe(Fall[sel], y[sel], seeds=(0,))
                rows.append({"frac": frac, "n_subjects": int(sel.sum()), "rep": rep,
                             "feature": name, "score": m})
        got = [r for r in rows if r["frac"] == frac]
        line = "  ".join(f"{nm}: {np.mean([r['score'] for r in got if r['feature'] == nm]):+.4f}"
                         for nm in feats)
        print(f"  n={got[0]['n_subjects']:4d} ({frac:.0%})  {line}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{'=' * 74}\nmean score by n ({'AUC' if binary else 'rho'}), {args.reps} subsamples each")
    piv = df.pivot_table(index="n_subjects", columns="feature", values="score")
    print(piv.round(4).to_string())
    print("\nslope over the LAST HALF of the curve (per +100 subjects):")
    for nm in feats:
        sub = piv[nm].dropna()
        h = sub.iloc[len(sub) // 2:]
        if len(h) > 1:
            sl = np.polyfit(h.index.values.astype(float), h.values, 1)[0] * 100
            verdict = "still rising" if sl > 0.005 else ("flat" if abs(sl) <= 0.005 else "declining")
            print(f"  {nm:26s} {sl:+.4f} per 100 subjects   -> {verdict}")
    print("  a flat tail means extra rows cannot help and the limit is signal, not n")
    print("=" * 74)

    out = Path(args.out) if args.out else (RUNS_DIR / "v2" / f"learning_curve_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(args), "rows": rows}, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
