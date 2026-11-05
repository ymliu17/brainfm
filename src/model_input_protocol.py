"""Unified regional time-series construction, before model-specific wrappers.

The primary recipe matches the existing ADNI ``amplitude=official`` build.
No cohort-dependent gain correction, ROI z-scoring, or extra signal regression.
"""
from dataclasses import replace
import torch

PROTOCOL = 'adni_official_unified_v2'


def build_input(x, length=200, *, strict=False, qc_policy="basic"):
    """Crop real samples, center each ROI, then append zero padding.

    ``x`` contains real frames only. Callers must strip storage padding using
    recorded valid lengths. Strict QC is retained for new external cohorts.
    """
    if x.ndim != 2 or x.shape[1] < 1 or length < 1:
        raise ValueError('invalid_shape_or_length')
    if qc_policy not in ("basic", "sparse"):
        raise ValueError("Unknown ROI QC policy")
    if qc_policy == "sparse":
        from roi_signal_qc import require_usable
        require_usable(x, length)
    x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite_real_frames')
    if strict and (x.std(1, unbiased=False) <= 1e-8).any():
        raise ValueError('constant_roi_full')
    y = x[:, :length]
    if strict and (y.std(1, unbiased=False) <= 1e-8).any():
        raise ValueError('constant_roi_window')
    y = y - y.mean(1, keepdim=True)
    return torch.nn.functional.pad(y, (0, max(0, length-y.shape[1]))).contiguous()


def build_cohort(co, length=200):
    """Apply the same recipe separately to each participant and visit."""
    out = torch.empty((co.bold.shape[0], co.bold.shape[1], length, co.bold.shape[3]), dtype=torch.float32)
    for i in range(out.shape[0]):
        for visit in range(out.shape[3]):
            n = int(co.valid[i, visit])
            if n < 1 or n > co.bold.shape[2]:
                raise ValueError('invalid_recorded_length')
            out[i, :, :, visit] = build_input(co.bold[i, :, :n, visit], length)
    return replace(co, bold=out, valid=co.valid.clone().clamp(max=length))
