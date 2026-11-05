#!/usr/bin/env python3
"""ADNI fingerprinting matched to the RCT run, to resolve a contradiction between them.

THE CONTRADICTION. §4 measured ADNI fingerprinting and found that identity is present at
the INPUT and shed with depth: `roi_var` peaks at 0.311 at layer 0 -- before a single
transformer block -- and decays to 0.179 at layer 4, and the best encoder cell (emb_fc L4,
0.209) sits BELOW every closed-form baseline (FC 0.221, amp 0.362, tucker-FC 0.426).
`scripts/fingerprint_rct.py` then measured the three trials and found the OPPOSITE in all
three: `roi_var` rises 0.421->0.874 (ACT), 0.696->0.885 (CogTE), 0.481->0.949 (BREATHE),
and on BREATHE the deep cell EXCEEDS every closed-form baseline.

Those two runs are not comparable, for three reasons at once, and until they are the
direction of the depth effect is a property of one run rather than of the encoder:

  window          §4 used 140 frames; the RCT run uses the official 200
  normalisation   §4 went through `dapt_adni.AdniWindows`; the RCT run through
                  `cohorts.center_rois` and the prepared build
  n               452 subjects against 74-107, and top1 is a rank-1 hit rate against n-1
                  distractors, so it is not comparable across n at fixed representation
                  quality

This script removes all three. It reads the PREPARED t200 ADNI build -- the same tensors,
the same per-ROI mean-centring and the same locked zero-pad policy that
`mci_axis_transfer.py` and the RCT cohorts use -- groups sessions by subject, and reports
top1 at full n AND at each RCT cohort's n by subsampling. Same families, same readouts,
same metric function as both earlier runs.

THE PADDING CONTROL IS NOT OPTIONAL HERE. 707 of 994 sessions in the DX build are shorter
than 200 frames, so ADNI at the official length is far more heavily padded than the trials
were (where only CogTE needed it). Zero-padding is a per-session property and a
participant's two sessions may share a length, which would hand the matcher a free cue. So
`scan length` is carried as its own feature family, built from the count of NONZERO frames
actually present in the tensor rather than from a manifest column, because the tensor is
what the encoder sees. Any cell that does not clearly exceed that control is uninterpretable.

Labels are irrelevant to fingerprinting, so the build with the most repeat sessions is used:
`raw_ADNI_MEM_all_t200_off` has 1658 sessions over 718 subjects, 437 of them with two or
more, against 273 in the DX build.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config as C  # noqa: E402
from datasets import load_xyz  # noqa: E402
from fingerprint_adni import READOUTS, fingerprint  # noqa: E402
from fingerprint_rct import subsampled  # noqa: E402
from paths import DERIVED_DIR, OFFICIAL_BRAINLM_13M, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from benchmark_gpu_phase import acquire_gpu_for_stage as gpu_acquire, release_gpu_for_cpu_tail as gpu_release, gpu_call

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(DERIVED_DIR / "adni424_t200"))
    ap.add_argument("--tag", default="raw_ADNI_MEM_all_t200_off")
    ap.add_argument("--tlen", type=int, default=200)
    ap.add_argument("--layers", default="0,1,2,3,4")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_subs", type=int, nargs="+", default=[74, 95, 107],
                    help="RCT cohort sizes, for the matched-n columns")
    ap.add_argument("--n_sub_reps", type=int, default=50)
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--out", default="runs/v2/fingerprint_adni_matched.json")
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.tlen
    layers = [int(x) for x in a.layers.split(",")]
    xyz = load_xyz(XYZ_PATH)

    import pandas as pd
    from probe_readouts import extract
    from thirdorder import shared_coskew_basis, tucker_fc_feature
    from transfer_baseline import amp_feature, fc_feature

    x = torch.load(f"{a.dir}/ADNI_x_{a.tag}.pt", map_location="cpu",
                   weights_only=False).float()
    man = pd.read_csv(f"{a.dir}/ADNI_manifest_{a.tag}.csv")
    # Effective length from the TENSOR, not the manifest: the manifest records the raw
    # acquisition, while what the encoder sees is the resampled, padded window.
    nzf = (x.abs().sum(1) > 0).sum(1).numpy()

    by = defaultdict(list)
    for i, p in enumerate(man["PTID"].values):
        by[p].append(i)
    subs = sorted(s for s, v in by.items() if len(v) >= 2)
    ia = np.array([by[s][0] for s in subs])
    ib = np.array([by[s][1] for s in subs])
    A, B = x[ia][:, :, :a.tlen].contiguous(), x[ib][:, :, :a.tlen].contiguous()
    n = len(subs)
    same_len = float((nzf[ia] == nzf[ib]).mean())
    print(f"{len(man)} sessions over {len(by)} subjects -> {n} with two sessions")
    print(f"chance top1 {1 / n:.5f}   sessions padded: "
          f"{int((nzf[ia] < a.tlen).sum() + (nzf[ib] < a.tlen).sum())}/{2 * n}")
    print(f"the two sessions share a length for {same_len:.1%} of subjects "
          f"(RCT: 28% ACT, 43% CogTE, 38% BREATHE)", flush=True)

    fam = {}
    an, bn = A.numpy(), B.numpy()
    fam["FC edges"] = (fc_feature(an, None), fc_feature(bn, None))
    fam["amp profile"] = (amp_feature(an, None), amp_feature(bn, None))
    basis = shared_coskew_basis(an, None, rank=64)
    fam["tucker-FC"] = (tucker_fc_feature(an, None, rank=64, basis=basis),
                        tucker_fc_feature(bn, None, rank=64, basis=basis))
    fam["scan length (CONTROL)"] = (
        np.column_stack([nzf[ia], nzf[ia] ** 2]).astype(float),
        np.column_stack([nzf[ib], nzf[ib] ** 2]).astype(float))

    from model_loader import load_brainlm_forpretraining
    gpu_acquire()
    m = load_brainlm_forpretraining(Path(a.pretrained)).to(DEVICE).eval()
    for li in layers:
        fa = extract(m, A, xyz, [li], list(READOUTS), a.batch_size)
        fb = extract(m, B, xyz, [li], list(READOUTS), a.batch_size)
        for r in READOUTS:
            fam[f"BrainLM 13M pretrained | {r} L{li}"] = (fa[(li, r)], fb[(li, r)])
        print(f"  layer {li} done", flush=True)
    del m
    torch.cuda.empty_cache()
    gpu_release()

    rows = []
    hdr = "".join(f"{f'n={k}':>9}" for k in a.n_subs)
    print(f"\n  {'family':<40}{'dim':>8}{'top1':>8}{'xchance':>9}{hdr}{'I_diff':>9}{'z_self':>8}")
    for k, (FA, FB) in fam.items():
        FA, FB = np.asarray(FA, float), np.asarray(FB, float)
        d = fingerprint(FA, FB)
        subvals = {}
        for ns in a.n_subs:
            s, _ = subsampled(FA, FB, ns, a.n_sub_reps)
            subvals[ns] = s
        rows.append(dict(family=k, dim=int(FA.shape[1]), n_subjects=n, chance=1.0 / n,
                         top1_sub={str(kk): vv for kk, vv in subvals.items()}, **d))
        print(f"  {k:<40}{FA.shape[1]:>8}{d['top1']:>8.3f}{d['top1'] * n:>8.0f}x"
              + "".join(f"{subvals[ns]:>9.3f}" for ns in a.n_subs)
              + f"{d['I_diff']:>9.4f}{d['z_self']:>8.2f}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"args": vars(a), "n_subjects": n, "chance": 1.0 / n,
         "frac_sessions_same_length": same_len, "rows": rows}, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
