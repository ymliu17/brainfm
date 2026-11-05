"""Outcome-independent signal-availability policy for every benchmark cohort.

Sparse constant parcels are retained without imputation. This numerical audit
does not establish anatomical coverage, registration quality or motion quality.
"""
import torch

POLICY = 'sparse_roi_v1'
SD_EPS = 1e-6
MAX_DEAD_FRACTION = .05


def assess(x, length):
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 2 or length < 2:
        raise ValueError('invalid_shape_or_length')
    x = x.float()
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite_real_frames')
    full = x.std(1, unbiased=False) < SD_EPS
    window = x[:, :length].std(1, unbiased=False) < SD_EPS
    dead = full | window
    count = int(dead.sum())
    return dict(policy=POLICY, n_roi=len(x), n_dead_full=int(full.sum()),
                n_dead_window=int(window.sum()), n_dead=count,
                dead_fraction=count / len(x), strict_pass=count == 0,
                primary_pass=count <= int(MAX_DEAD_FRACTION * len(x)))


def require_usable(x, length):
    q = assess(x, length)
    if not q['primary_pass']:
        raise ValueError(f"extensive_constant_rois:{q['n_dead']}/{q['n_roi']}")
    return q
