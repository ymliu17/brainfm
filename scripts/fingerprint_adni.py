#!/usr/bin/env python3
"""Can a representation recognise the same subject in a second scan?

This is the direct measurement of the quantity `dapt_recon_diag.py` implicated. That script
established that continued pretraining on ADNI makes BrainLM a genuinely better
reconstructor -- 6.1x lower masked MSE, recovering 0.761 of the variance a subject-generic
covariance cannot explain -- while the downstream probe does not move (20 identical cells,
median difference exactly 0.0000). The proposed explanation was that masked reconstruction
at mask_ratio 0.2 is a CONDITIONAL INTERPOLATION task: 80% of the tokens are visible, so
subject-specific structure can reach the decoder through the visible tokens at inference
time and never has to be compressed into a summary. A frozen-encoder probe reads exactly
such a summary.

Fingerprinting tests that directly and without any label. Take the subjects with two
sessions, compute one feature vector per session, and ask whether session A's nearest
neighbour among all session Bs is the same person (Finn et al. 2015; Amico & Goni 2018 for
the differential-identifiability form). A representation that compresses a subject well
scores high whatever the subject's diagnosis; one that only supports interpolation scores
at chance. The prediction under the explanation above: FC high, both BrainLM checkpoints
low, and adaptation not helping.

Reported per feature family:
  top1        mean of the two directions, A->B and B->A. Chance is 1/n_subjects.
  I_diff      mean same-subject similarity minus mean different-subject similarity, the
              scale-free version of the same question (Amico & Goni).
  z_self      how many between-subject SDs the true match sits above the crowd, averaged
              over subjects. Insensitive to the overall similarity level, which differs a
              lot between a 424-dim and an 89676-dim feature.

Windows are the centre 140 frames with the per-ROI mean-centring and single-scalar gain
that `dapt_adni.AdniWindows` applies, so the encoders see exactly what they were adapted
and diagnosed on. Every kept session is at least 140 frames long, so the LOCKED zero-pad
policy for short scans never engages here and cannot confound the comparison.

ANSWER (2026-09-02, runs/v2/fingerprint_adni.json, n=452 subjects, chance top1 0.0022):

  family                  best cell     top1   I_diff  z_self
  tucker-FC               dim 2016     0.426   0.0980    3.65
  amp profile             dim 424      0.362   0.1677    2.10
  BrainLM 13M +DAPT       emb_fc L4    0.340   0.0770    1.56
  BrainLM 13M pretrained  roi_var L0   0.311   0.1603    1.91
  FC edges                dim 89676    0.221   0.1189    2.05

  top1 by depth              L0    L1    L2    L3    L4
  cls          pretrained  .002  .002  .011  .006  .001
               +DAPT       .002  .002  .009  .015  .009
  roi_var      pretrained  .311  .233  .247  .118  .179
               +DAPT       .305  .254  .283  .227  .270
  emb_fc       pretrained  .024  .084  .092  .192  .209
               +DAPT       .024  .071  .090  .291  .340

Two findings, and the second one refutes the explanation this script was written to test.

First, the pooled readouts are EMPTY. `cls` sits at 0.002-0.015 against a 0.0022 chance
line in both checkpoints, and `mean` at 0.008-0.028. The canonical way to read a
transformer cannot tell 452 people apart at all. Only per-ROI magnitude (`roi_var`,
`roi_norm`) and a covariance taken over the embeddings (`emb_fc`) identify anyone -- and
`roi_var` is already at 0.311 at layer 0, i.e. before a single transformer block, then
DECAYS with depth to 0.179. The pretrained encoder discards subject identity as it goes.

Second, adaptation DOES make the representation more identifiable: 10 of the 15
non-degenerate cells improve, mean +0.051, and the deep cells move a lot (emb_fc L4
0.209 -> 0.340, roi_norm L3 0.114 -> 0.253). It also stops the decay with depth. So
`dapt_recon_diag.py`'s conclusion -- that adaptation failed to make subject-specific
information accessible from a fixed summary -- is wrong. It did.

What did not move, at all, is the diagnosis: 20 identical (readout, layer) cells on ADNI
CN/AD, adapted wins 9/20, median difference exactly 0.0000.

So the real result is a DISSOCIATION. Reconstruction quality (6.1x), subject-specific
reconstruction (resid_ve -0.415 -> 0.761) and subject identifiability (+0.051) all move
together under adaptation, and clinical predictivity moves not at all. Identifiability is
not a proxy for predictivity.

The baselines show the same dissociation independently, which is the reason to believe it
is a property of the data and not of the encoder: `amp profile` fingerprints BETTER than
`FC edges` (0.362 vs 0.221) and predicts CN/AD WORSE (0.731 vs 0.797). What makes a
subject identifiable in fMRI -- stable per-ROI amplitude and SNR structure, which tracks
head size, registration residual and vascular anatomy -- is maximally reliable and
therefore the first thing any reconstruction objective encodes, and it carries no disease
information. That last clause is interpretation, not measurement.

It also resolves the apparent conflict with the reliability table
(runs/v2/axis_sensitivity*.json): a BrainLM axis drifts by 0.71 of the CN-AD gap between
sessions while the representation it is drawn from identifies subjects at 100x chance. The
representation is stable; the disease-relevant DIRECTION inside it is weak, so a fitted
1-D projection lands mostly on components that do not replicate.

Prescription: stop reporting reconstruction loss, and do not substitute fingerprinting for
it. Both move independently of the only quantity that matters here.
"""
from __future__ import annotations

import argparse
import json
import re
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
from paths import ADNI_SCAN_DIR, OFFICIAL_BRAINLM_13M, RUNS_DIR, XYZ_PATH  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUBJECT_RE = re.compile(r"(sub-ADNI\d+S\d+)_")
READOUTS = ("cls", "mean", "roi_var", "roi_norm", "emb_fc")


def fingerprint(FA, FB):
    """Identification statistics from two [N, d] matrices of paired sessions.

    Similarity is the Pearson correlation across feature dimensions, which removes each
    session's own offset and scale. That matters here: the families being compared range
    from 424 log-amplitudes to 89676 Fisher-z edges, and an inner product would rank them
    by norm rather than by how well they identify anyone.
    """
    def z(F):
        F = F - F.mean(1, keepdims=True)
        return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    S = z(np.nan_to_num(FA)) @ z(np.nan_to_num(FB)).T          # [N, N]
    n = len(S)
    same = np.diag(S).copy()
    off = S.copy()
    np.fill_diagonal(off, np.nan)
    a = float((S.argmax(1) == np.arange(n)).mean())
    b = float((S.argmax(0) == np.arange(n)).mean())
    mu = np.nanmean(off, axis=1)
    sd = np.maximum(np.nanstd(off, axis=1), 1e-12)
    return {"top1": (a + b) / 2, "top1_ab": a, "top1_ba": b,
            "I_diff": float(same.mean() - np.nanmean(off)),
            "z_self": float(np.mean((same - mu) / sd)), "n": n}


def build_pairs(scan_dir, window, seed):
    """Two QC-passing sessions per subject, as [N, 424, window] tensors.

    QC and normalisation are `AdniWindows`' own, so a session excluded from the adaptation
    is excluded here too and the encoders are never handed a scan they were never meant to
    see.
    """
    from dapt_adni import AdniWindows
    files = sorted(Path(scan_dir).glob("*.pt"))
    ds = AdniWindows(files, window, seed=seed, deterministic=True)
    by = defaultdict(list)
    for i, f in enumerate(ds.kept):
        by[SUBJECT_RE.match(f.name).group(1)].append(i)
    subs = sorted(s for s, v in by.items() if len(v) >= 2)
    A = torch.stack([ds[by[s][0]] for s in subs])
    B = torch.stack([ds[by[s][1]] for s in subs])
    print(f"{len(ds)} QC-passing sessions -> {len(subs)} subjects with two "
          f"({len(by)} subjects total)", flush=True)
    return A, B, subs


def baseline_feats(A, B):
    from thirdorder import shared_coskew_basis, tucker_fc_feature
    from transfer_baseline import amp_feature, fc_feature
    a, b = A.numpy(), B.numpy()
    out = {"FC edges": (fc_feature(a, None), fc_feature(b, None)),
           "amp profile": (amp_feature(a, None), amp_feature(b, None))}
    # The co-skewness basis is fitted on session A only and applied to both. Pooling both
    # sessions would be unsupervised and so not label leakage, but it would let the basis
    # be shaped by the very scans it is asked to match.
    basis = shared_coskew_basis(a, None, rank=64)
    out["tucker-FC"] = (tucker_fc_feature(a, None, rank=64, basis=basis),
                        tucker_fc_feature(b, None, rank=64, basis=basis))
    return out


def encoder_feats(ckpt, A, B, xyz, layers, batch):
    """{(layer, readout): (FA, FB)} for one checkpoint, one layer per pass.

    Extracting all layers in a single pass would hold 5 x 2N x 89676 floats for `emb_fc`
    alone; looping layers keeps that to one layer at a time at the cost of re-running the
    encoder, which is cheap for a 13M model.
    """
    from model_loader import load_brainlm_forpretraining
    from probe_readouts import extract
    m = load_brainlm_forpretraining(ckpt).to(DEVICE).eval()
    out = {}
    for li in layers:
        fa = extract(m, A, xyz, [li], list(READOUTS), batch)
        fb = extract(m, B, xyz, [li], list(READOUTS), batch)
        for r in READOUTS:
            out[(li, r)] = (fa[(li, r)], fb[(li, r)])
        print(f"  layer {li} done", flush=True)
    del m
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan_dir", default=str(ADNI_SCAN_DIR))
    ap.add_argument("--pretrained", default=str(OFFICIAL_BRAINLM_13M))
    ap.add_argument("--adapted", default=str(RUNS_DIR / "v2/dapt/dapt_off_w140.pth"))
    ap.add_argument("--window", type=int, default=140)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layers", default="0,1,2,3,4")
    ap.add_argument("--out", default=str(RUNS_DIR / "v2/fingerprint_adni.json"))
    a = ap.parse_args()
    C.configure_backends()
    C.NUM_TIMEPOINTS_PER_VOXEL = a.window
    layers = [int(x) for x in a.layers.split(",")]

    A, B, subs = build_pairs(a.scan_dir, a.window, a.seed)
    rows = []
    for name, (fa, fb) in baseline_feats(A, B).items():
        rows.append({"family": name, "cell": f"dim {fa.shape[1]}", **fingerprint(fa, fb)})
        print(f"  {name:<14} top1 {rows[-1]['top1']:.3f}", flush=True)

    xyz = load_xyz(XYZ_PATH).to(DEVICE)
    for label, ck in (("BrainLM 13M pretrained", a.pretrained),
                      ("BrainLM 13M +DAPT", a.adapted)):
        if not Path(ck).exists():
            print(f"!! {ck} missing, skipped")
            continue
        print(f"\n{label}")
        for (li, r), (fa, fb) in encoder_feats(ck, A, B, xyz, layers, a.batch_size).items():
            rows.append({"family": label, "cell": f"{r} L{li}", **fingerprint(fa, fb)})

    chance = 1.0 / len(subs)
    print(f"\n{'=' * 78}")
    print(f"subject identification from a second session, n={len(subs)} "
          f"(chance top1 = {chance:.4f})")
    print("=" * 78)
    print(f"  {'family':<24}{'best cell':<12}{'top1':>8}{'I_diff':>9}{'z_self':>8}")
    fams = []
    for f in dict.fromkeys(r["family"] for r in rows):
        best = max((r for r in rows if r["family"] == f), key=lambda r: r["top1"])
        fams.append(best)
        print(f"  {f:<24}{best['cell']:<12}{best['top1']:>8.3f}"
              f"{best['I_diff']:>9.4f}{best['z_self']:>8.2f}")
    print("\n  every encoder cell, top1:")
    for f in ("BrainLM 13M pretrained", "BrainLM 13M +DAPT"):
        cells = [r for r in rows if r["family"] == f]
        if not cells:
            continue
        print(f"    {f}")
        for r in READOUTS:
            line = "".join(f"{c['top1']:>8.3f}" for c in cells if c["cell"].startswith(r + " "))
            print(f"      {r:<9}{line}   (L{', L'.join(map(str, layers))})")
    Path(a.out).write_text(json.dumps({"args": vars(a), "chance": chance,
                                       "n_subjects": len(subs), "rows": rows}, indent=2))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
