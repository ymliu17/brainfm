"""T1/T2 dual-timepoint fMRI datasets and the temporal geometry applied on load."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from temporal import PATCH, resample_time, trim_to_patch, zscore_time


def load_xyz(xyz_path: str | Path) -> torch.Tensor:
    xyz_arr = np.loadtxt(xyz_path, usecols=(1, 2, 3))
    return torch.tensor(xyz_arr, dtype=torch.float32)


def valid_lengths(x: torch.Tensor) -> torch.Tensor:
    """Usable frames per scan in a NaN-padded [N, R, T, 2] tensor -> [N, 2] long.

    The 2026-08 cohort tensors are padded with NaN out to the longest scan in the
    cohort, because motion censoring leaves each run a slightly different length. Two
    properties are asserted rather than assumed, since a violation would mean the
    padding is not what it looks like and every downstream frame count would be wrong:
    the NaN mask is identical across ROIs, and the NaN frames form one trailing block.
    Tensors with no NaN at all return the full length.
    """
    nan = torch.isnan(x)
    if not nan.any():
        return torch.full((x.shape[0], x.shape[3]), x.shape[2], dtype=torch.long)
    if not torch.equal(nan, nan[:, :1].expand_as(nan)):
        raise ValueError(
            "NaN mask differs between ROIs -- this is not whole-frame padding, so it "
            "cannot be handled by cropping. Inspect the tensor before training on it."
        )
    mask = ~nan[:, 0]                                       # [N, T, 2]
    v = mask.sum(dim=1)                                     # [N, 2]
    frame = torch.arange(mask.shape[1]).view(1, -1, 1)
    if not torch.equal(mask, frame < v.unsqueeze(1)):
        raise ValueError("NaN frames are not a single trailing block per scan")
    return v.long()


def apply_temporal_spec(x: torch.Tensor, native_tr: float | None, target_tr: float | None,
                        tlen: int | None, patch_size: int = PATCH,
                        valid: torch.Tensor | None = None) -> torch.Tensor:
    """Resample / crop / re-standardise a [N, R, T, 2] tensor along time.

    Kept here rather than in a preprocessing script so the pipeline can be pointed at
    the original `data/rct/` tensors and still train on the configured temporal
    geometry -- no parallel copies of the data to keep in sync. `features.py` folds the
    same parameters into the cache key.

    `valid` (from `valid_lengths`) makes the crop per scan: each run is cut to its own
    usable length *before* resampling, so a censored run contributes only real frames
    and never the NaN padding. `tlen` is then required, because scans of different
    lengths have to end up on one common grid.
    """
    if valid is not None and int(valid.min()) != int(valid.max()):
        return _apply_per_scan(x, native_tr, target_tr, tlen, patch_size, valid)
    if valid is not None:
        x = x.narrow(2, 0, int(valid.min()))
    if target_tr and native_tr and abs(native_tr - target_tr) > 1e-9:
        x = resample_time(x, native_tr, target_tr, dim=2)
    if tlen:
        if x.shape[2] < tlen:
            raise ValueError(f"need {tlen} frames along time, tensor has {x.shape[2]}")
        x = x.narrow(2, 0, tlen)
    if target_tr or tlen or valid is not None:
        # Resampling and cropping both break the per-ROI z-scoring the tensors ship with.
        x = zscore_time(trim_to_patch(x, dim=2, patch=patch_size), dim=2)
    return x


def _apply_per_scan(x, native_tr, target_tr, tlen, patch_size, valid):
    """Scans have different usable lengths, so crop each one before resampling."""
    if not tlen:
        raise ValueError(
            f"usable length varies across scans ({int(valid.min())}-{int(valid.max())} "
            "frames), so a common tlen is required; see temporal.COHORT_RESAMPLE"
        )
    if tlen % patch_size:
        raise ValueError(f"tlen={tlen} is not a multiple of the patch size {patch_size}")
    resample = bool(target_tr and native_tr and abs(native_tr - target_tr) > 1e-9)
    out = x.new_empty((x.shape[0], x.shape[1], tlen, x.shape[3]))
    for tp in range(x.shape[3]):
        v = valid[:, tp]
        for length in sorted(set(v.tolist())):
            idx = (v == length).nonzero(as_tuple=True)[0]
            blk = x[idx][:, :, :length, tp]
            if resample:
                blk = resample_time(blk, native_tr, target_tr, dim=2)
            if blk.shape[2] < tlen:
                raise ValueError(
                    f"a scan with {length} usable frames yields {blk.shape[2]} after "
                    f"resampling, short of tlen={tlen}"
                )
            out[idx, :, :, tp] = zscore_time(blk[:, :, :tlen], dim=2)
    return out


class T12DualDataset(Dataset):
    """Holds [N, 424, T, 2] BOLD plus labels, on a patch-aligned temporal grid.

    Accepts either file paths or in-memory tensors (`cohorts.load_cohort` supplies the
    latter, having already joined imaging to labels by subject ID). `native_tr` /
    `target_tr` / `tlen` apply the cohort's configured temporal geometry, and `valid`
    crops away NaN padding per scan; see temporal.COHORT_RESAMPLE.
    """

    def __init__(self, x: str | Path | torch.Tensor, y: str | Path | torch.Tensor,
                 xyz: torch.Tensor, patch_size: int = PATCH,
                 native_tr: float | None = None, target_tr: float | None = None,
                 tlen: int | None = None, valid: torch.Tensor | None = None):
        if not torch.is_tensor(x):
            x = torch.load(x, map_location="cpu", weights_only=False)
        if not torch.is_tensor(y):
            y = torch.load(y, map_location="cpu", weights_only=False)
        assert x.shape[0] == y.shape[0], "sample count mismatch between X and y"
        if valid is None:
            valid = valid_lengths(x)
        x = apply_temporal_spec(x.float(), native_tr, target_tr, tlen, patch_size, valid)
        pad_len = (patch_size - x.shape[2] % patch_size) % patch_size
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        if torch.isnan(x).any():
            raise ValueError("NaN survived temporal preprocessing")
        self.t1 = x[..., 0]
        self.t2 = x[..., 1]
        self.y = y.float()
        self.xyz = xyz

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.t1[idx], self.t2[idx], self.y[idx], self.xyz, idx
