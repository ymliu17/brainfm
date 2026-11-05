"""Functional-connectivity (FC) features for the SVM / MLP / GCN baselines.

Manuscript (Compared ML and DL methods):
  "For each participant, FC matrices were computed separately for T1 and T2 rsfMRI
   scans. The upper triangular elements of each FC matrix were flattened into
   feature vectors and concatenated across timepoints to explicitly encode
   longitudinal change."

Input tensors follow the same layout as the FM pipeline: X is [N, R, T, 2] with the
last dim = (T1, T2) and R = 424 A424 ROIs.

Since the 2026-08 refresh those tensors are NaN-padded out to the cohort's longest scan,
so every function here takes the per-scan usable length. Without it `np.corrcoef` returns
all-NaN and `nan_to_num` quietly turns the whole connectome into zeros -- a baseline that
scores at chance for a reason that has nothing to do with connectivity.
"""
from __future__ import annotations

import numpy as np
import torch


def compute_fc(timeseries: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Pearson FC per subject.

    Args:
        timeseries: [N, R, T] BOLD signal (single timepoint scan).
        valid: [N] usable frames per scan; the rest is NaN padding and is dropped.
            None means every scan uses all T frames.
    Returns:
        [N, R, R] correlation matrices.
    """
    ts = np.asarray(timeseries, dtype=np.float64)
    n, r, t = ts.shape
    fc = np.empty((n, r, r), dtype=np.float64)
    for i in range(n):
        run = ts[i, :, : int(valid[i])] if valid is not None else ts[i]
        if not np.isfinite(run).all():
            raise ValueError(
                f"scan {i} still holds non-finite values over its first "
                f"{run.shape[1]} frames -- pass the correct `valid` lengths"
            )
        # np.corrcoef over rows (ROIs); guard against zero-variance ROIs.
        c = np.corrcoef(run)
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        fc[i] = c
    return fc


def upper_tri(fc: np.ndarray) -> np.ndarray:
    """Flatten strict upper triangle (excluding diagonal).

    Args:
        fc: [N, R, R]
    Returns:
        [N, R*(R-1)/2]
    """
    r = fc.shape[1]
    iu = np.triu_indices(r, k=1)
    return fc[:, iu[0], iu[1]]


def fc_features_t1t2(x: torch.Tensor | np.ndarray, valid=None) -> np.ndarray:
    """Concatenated [upper-tri FC(T1) ; upper-tri FC(T2)] feature vector.

    Args:
        x: [N, R, T, 2] dual-timepoint tensor.
        valid: [N, 2] usable frames per scan (see compute_fc).
    Returns:
        [N, 2 * R*(R-1)/2] feature matrix.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    v = _valid_cols(valid)
    f1 = upper_tri(compute_fc(x[..., 0], v[0]))
    f2 = upper_tri(compute_fc(x[..., 1], v[1]))
    return np.concatenate([f1, f2], axis=1)


def fc_matrices_t1t2(x: torch.Tensor | np.ndarray, valid=None):
    """Return the raw FC matrices (T1, T2) — used by the GCN node features.

    Args:
        x: [N, R, T, 2]
        valid: [N, 2] usable frames per scan (see compute_fc).
    Returns:
        (fc_t1, fc_t2): each [N, R, R]
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    v = _valid_cols(valid)
    return compute_fc(x[..., 0], v[0]), compute_fc(x[..., 1], v[1])


def _valid_cols(valid):
    """Split a [N, 2] usable-length array into (T1, T2), or (None, None)."""
    if valid is None:
        return None, None
    if isinstance(valid, torch.Tensor):
        valid = valid.cpu().numpy()
    valid = np.asarray(valid)
    return valid[:, 0], valid[:, 1]
