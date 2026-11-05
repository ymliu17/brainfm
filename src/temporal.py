"""Temporal resampling, so a BrainLM token means the same thing everywhere.

BrainLM was pretrained on UK Biobank alone (TR 0.735 s) with
`timepoint_patching_size=20`, so one token spanned ~14.7 s of real time. (HCP appears in
the BrainLM paper as a downstream EVALUATION set, not in pretraining; its TR of 0.72 s is
close enough that the distinction does not change the target grid below.) The encoder's
signal projection is `Linear(20, hidden)`: the patch size is baked into the pretrained
weights, so it cannot be changed to suit a new TR. The only lever is the sampling grid.

What a token spans in this project at native TR:

    ADNI3 advanced   TR 0.607 s -> 12.1 s   (0.82x pretraining)
    ACT              TR 1.03  s -> 20.6 s   (1.40x)
    CogTE            TR 2.5   s -> 50.0 s   (3.40x)
    ADNI2 / basic    TR 3.0   s -> 60.0 s   (4.08x)

Resampling to `PRETRAIN_TR` makes every token span 14.7 s. It cannot fix spectral
content: Nyquist is 0.485 Hz at TR 1.03 s, 0.200 Hz at 2.5 s and 0.167 Hz at 3.0 s,
against 0.68 Hz for the pretraining data. Upsampling interpolates; it does not recover
a band that was never acquired. ACT is the mild case (1.4x, 0.485 -> 0.68 Hz); CogTE
and the TR 3.0 s ADNI scans are not.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import torch
from scipy import signal

#: Effective repetition time of BrainLM's pretraining corpora (UKB 0.735 s, HCP 0.72 s).
PRETRAIN_TR = 0.735

#: Patch size the pretrained signal projection expects.
PATCH = 20

#: Known repetition times, seconds. Sources: cohort BIDS sidecars (ACT, CogTE) and the
#: ADNI acquisition protocols (length identifies the ADNI protocol; see adni_native_tr).
#:
#: BREATHE arrived with the 2026-08 refresh and nothing in the package documents its TR.
#: It is deliberately left None rather than guessed from its frame count (300, the same
#: as ACT): the TR sets how many seconds a token spans, so a wrong value silently
#: mis-scales every embedding. Supply it with --native_tr.
COHORT_TR = {"act": 1.03, "cogte": 2.5, "breathe": None, "psopt": 1.375}

#: Default temporal preprocessing per cohort, applied on load (see cohorts.load_cohort).
#:
#: What drives downstream performance is the number of temporal patches the frozen
#: encoder sees, not the duration of one token (the original cohort audit 2c). CogTE natively
#: carries 100/20 = 5 patches and sits at chance; resampling to 0.735 s and keeping 280
#: frames gives 14 patches and lifts `cogte_t1` pooled out-of-fold AUC from 0.5079 to
#: 0.6761. ACT natively has ~300/20 = 15 patches, already on the plateau, and resampling
#: it to 21 gains nothing while costing 1.4x the sequence length -- so ACT is not
#: resampled.
#:
#: `tlen` is pinned for every cohort. The 2026-08 tensors
#: are NaN-padded to the cohort's longest scan and each scan's usable length differs
#: (ACT 291-300, CogTE 95-118, BREATHE 295-300 frames), so "use the stored length" no
#: longer names one geometry. 280 frames = 14 patches is the largest patch-aligned
#: window every scan can supply: ACT/BREATHE natively, CogTE after resampling (its
#: shortest scan, 95 frames at TR 2.5 s, becomes 323 frames at 0.735 s).
#:
#: `target_tr=None` means "use the tensor's own sampling grid".
COHORT_RESAMPLE = {
    "act": {"target_tr": None, "tlen": 280},
    "cogte": {"target_tr": PRETRAIN_TR, "tlen": 280},
    "breathe": {"target_tr": None, "tlen": 280},
    # psopt is NOT resampled and is pinned to the same 280 as the others even though it
    # could supply 460 (its shortest session is 467 frames). The pin is deliberate: 280
    # keeps the token geometry identical across all four cohorts, so a psopt-vs-ACT
    # difference cannot be a sequence-length difference. psopt is the only cohort for which
    # 280 requires no padding at all -- 43/44 T1 and 44/44 T2 sessions hold the full 600
    # frames -- which is what makes it the control for the padding confound rather than
    # another instance of it.
    "psopt": {"target_tr": None, "tlen": 280},
}


def adni_native_tr(n_timepoints: int) -> float:
    """ADNI TR inferred from run length: >=750 volumes is the 0.607 s multiband protocol."""
    return 0.607 if n_timepoints >= 750 else 3.0


def token_seconds(tr: float, patch: int = PATCH) -> float:
    return tr * patch


def mismatch_ratio(tr: float, patch: int = PATCH) -> float:
    """How many times longer a token is than during pretraining."""
    return token_seconds(tr, patch) / token_seconds(PRETRAIN_TR, patch)


def nyquist(tr: float) -> float:
    return 0.5 / tr


def resample_time(x: torch.Tensor, tr_from: float, tr_to: float = PRETRAIN_TR,
                  dim: int = -1, max_denom: int = 200) -> torch.Tensor:
    """Resample along `dim` from tr_from to tr_to with polyphase FIR filtering.

    Handles [R, T], [N, R, T] and [N, R, T, 2] by moving `dim` last.
    """
    if abs(tr_from - tr_to) < 1e-9:
        return x
    arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    arr = np.moveaxis(arr, dim, -1)
    ratio = Fraction(tr_from / tr_to).limit_denominator(max_denom)
    out = signal.resample_poly(arr, ratio.numerator, ratio.denominator, axis=-1)
    out = np.moveaxis(out, -1, dim)
    return torch.from_numpy(np.ascontiguousarray(out)).float()



def resample_to_length(x: torch.Tensor, n_out: int, dim: int = -1) -> torch.Tensor:
    """Resample along `dim` so the length is exactly `n_out`, using the whole input.

    The alternative ways to make a short run reach a fixed length both waste something.
    Resampling it onto a canonical TR and cropping keeps the token duration the encoder
    was trained on but throws away most of the recording -- a 140-frame run at TR 3.0 s
    spans 420 s, and 200 frames at 0.735 s reads only the first 147 s of it. Zero-padding
    keeps the whole recording but spends 3 of 10 tokens on frames that are identical for
    every ROI and every subject, so attention still has to look at them. Stretching keeps
    all of the recording AND fills every token with real signal; what it gives up is that
    a token no longer spans a fixed number of seconds (42 s for a stretched 140-frame TR
    3.0 s run against 14.7 s in pretraining).

    That last cost is the one this project has actually measured. The `trhom` builds hold
    the token COUNT fixed and change only the duration, 60 s against 14.7 s, over 80
    paired probe cells: -0.0050 +/- 0.0151. Token duration is worth half an AUC point, so
    paying it to keep the entire recording and every token real is the better trade.

    Exact by construction: up/down are the target and source lengths over their gcd, so
    `resample_poly` lands on `n_out` rather than near it.
    """
    arr = x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    arr = np.moveaxis(arr, dim, -1)
    n_in = arr.shape[-1]
    if n_in == n_out:
        return x if isinstance(x, torch.Tensor) else torch.from_numpy(arr).float()
    g = np.gcd(n_in, n_out)
    out = signal.resample_poly(arr, n_out // g, n_in // g, axis=-1)
    if out.shape[-1] != n_out:                       # guard, not expected to fire
        raise ValueError(f"resample_poly gave {out.shape[-1]} frames, wanted {n_out}")
    out = np.moveaxis(out, -1, dim)
    return torch.from_numpy(np.ascontiguousarray(out)).float()

def trim_to_patch(x: torch.Tensor, dim: int = -1, patch: int = PATCH) -> torch.Tensor:
    """Drop trailing frames so the length is an exact multiple of the patch size."""
    n = x.shape[dim] // patch * patch
    return x.narrow(dim, 0, n)


def zscore_time(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Per-ROI z-score along time, matching how the shipped cohort tensors are stored."""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def describe(tr: float, n_timepoints: int) -> str:
    return (
        f"TR={tr:g}s  {n_timepoints} frames = {tr * n_timepoints:.0f}s  "
        f"token={token_seconds(tr):.1f}s ({mismatch_ratio(tr):.2f}x pretraining)  "
        f"nyquist={nyquist(tr):.3f}Hz"
    )
