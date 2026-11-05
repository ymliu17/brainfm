"""BrainLM's pretraining input, transcribed from the code that produced it.

Source: `toolkit/BrainLM_Toolkit.py` and `train.py` in
https://github.com/vandijklab/BrainLM. The 13M pretrains on the dataset column
`Voxelwise_RobustScaler_Normalized_Recording`, which `BrainLM_Toolkit.py` builds like this
(lines 297-298):

    recording_mean_subtracted3 = recording_mean_subtracted3 - recording_mean_subtracted3.mean(axis=0)
    recording_mean_subtracted3 = (recording_mean_subtracted3 - data_median_per_voxel / IQR)

**Read the second line carefully.** Python binds `/` tighter than `-`, so it computes
`x - (median / IQR)`, not `(x - median) / IQR`. Despite the column's name the data is never
divided by the IQR: only a per-voxel dataset-level *constant* is subtracted. Verified
numerically -- the per-voxel standard deviation after the second line is identical to the
one after the first.

So the distribution the 13M encoder actually learned on is:

    1. per-ROI mean-centring **within each recording** -- amplitude preserved
    2. minus a per-ROI constant computed once over the whole dataset (median / IQR)
    3. windows of 200 frames at TR 0.735 s (200 / 20 = 10 patches per ROI, seq len 4241)

That is the operative description, whether or not line 298 was a typo: these are the weights
that exist, and they were fitted on what the code produced.

`max_val = 5.6430855` is **not part of this path.** It appears only in the ViTMAE (111M /
650M) tutorials, where the recording is turned into an image and padded with the constant
-1; dividing by it puts the signal inside [-1, 1] so the padding stays distinguishable from
real data. See `src/vitmae_input.py`. `train.py` never references it.

## The amplitude mismatch, and how it was closed

The superseded `denoise-old` release (deleted 2026-08-25) was per-ROI z-scored **within
each scan**: `(x - mean) / sd`, with
per-ROI sd measured at exactly 1.000 for every ROI of every scan. Step 1 above only removes
the mean, so the mismatch was one specific thing -- **amplitude**. Between-ROI and
between-subject amplitude differences exist in the pretraining distribution and had been
divided out of ours.

`data/adni/raw` (2026-08-24) is the same recordings **before** that division; z-scoring it
reproduced the old release to 1.9e-6, which is why the old one was not worth keeping. Build
from it with
`prepare_adni.py --scan_dir data/adni/raw --amplitude preserve`, which mean-centres per ROI
and divides the whole scan by one scalar -- matching step 1 up to a global gain, which step 2
then absorbs. See data/README.md for the release's provenance and QC.

The quantity that was missing is reliable rather than noise: the per-scan amplitude profile
`sd / median(sd)` correlates at +0.593 within subject vs +0.438 between, and identifies the
subject 55% of the time out of 120 (chance 0.8%).

Note that FC is invariant to per-ROI rescaling (Pearson correlation is scale-free), so the
FC baselines are unaffected by this and the foundation models are not -- which was the
leading explanation for why FC out-predicts every BrainLM release on ADNI demographics
(the original cohort audit). Because FC is identical across the two builds (verified 2.7e-08), it
serves as an invariant reference that makes the amplitude-vs-z-scored comparison a clean
one-variable test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: BrainLM's pretraining TR (UK Biobank).
PRETRAIN_TR = 0.735

#: BrainLM's pretraining window, in frames. 200 / 20 = 10 time patches per ROI.
PRETRAIN_WINDOW = 200

#: Patch size baked into the pretrained signal projection.
PATCH = 20

#: Tolerance before resampling, matching the reference implementations.
TR_TOLERANCE = 0.1


def load_a424_coords(xyz_path: str | Path) -> np.ndarray:
    """A424 coordinates as [424, 3] in (X, Y, Z); column 0 of the file is the index."""
    arr = np.loadtxt(xyz_path, dtype=np.float32)
    return arr[:, 1:4]


def roi_order_by_y(xyz_path: str | Path) -> np.ndarray:
    """Indices that sort ROIs by Y coordinate.

    Needed by the ViTMAE releases, where a 2D convolution tiles neighbouring ROIs into one
    patch. Irrelevant for the 13M, where each (ROI, time-patch) is its own token.
    """
    return np.argsort(load_a424_coords(xyz_path)[:, 1], kind="stable")


def resample_linear(x: torch.Tensor, tr_from: float, tr_to: float = PRETRAIN_TR) -> torch.Tensor:
    """Linear temporal interpolation on [R, T].

    `src/temporal.resample_time` uses polyphase FIR instead, which is the better
    signal-processing choice; both are available so the difference can be measured rather
    than assumed away.
    """
    if abs(tr_from - tr_to) < TR_TOLERANCE:
        return x
    n_out = round(float(tr_from) * x.shape[-1] / float(tr_to))
    return F.interpolate(x.unsqueeze(0), size=n_out, mode="linear", align_corners=False).squeeze(0)


@dataclass
class AmplitudeStats:
    """Per-ROI standard deviations that undo the within-scan z-scoring (step 1).

    `source` records where they came from, because it decides whether a downstream result
    is on-distribution or an approximation:

      "measured"  the sd actually used to z-score this data. Restores the pretraining
                  distribution exactly. This is what to supply.
      "surrogate" sd taken from somewhere else (another dataset, a published atlas of ROI
                  variances). Restores *a* plausible amplitude profile, not this data's.
      "none"      no restoration; the input stays z-scored and off-distribution. Every
                  result computed this way must say so.
    """

    sd: torch.Tensor        # [R]
    source: str

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.source == "none":
            return x
        return x * self.sd.to(x.dtype).unsqueeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save({"sd": self.sd, "source": self.source}, path)

    @classmethod
    def load(cls, path: str | Path) -> "AmplitudeStats":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["sd"], d["source"])

    @classmethod
    def none(cls, n_roi: int = 424) -> "AmplitudeStats":
        return cls(torch.ones(n_roi), "none")


@dataclass
class VoxelOffset:
    """The per-ROI dataset constant subtracted in step 2: median / IQR.

    Computed once over a whole dataset, not per scan. The pretraining constant came from
    UK Biobank; fitting one on a different dataset reproduces the *operation* but not the
    same numbers, so `source` records which.
    """

    offset: torch.Tensor    # [R]
    source: str

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.offset.to(x.dtype).unsqueeze(-1)

    def save(self, path: str | Path) -> None:
        torch.save({"offset": self.offset, "source": self.source}, path)

    @classmethod
    def load(cls, path: str | Path) -> "VoxelOffset":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["offset"], d["source"])


def fit_voxel_offset(scans: list[torch.Tensor], source: str,
                     frames_per_scan: int | None = 200) -> VoxelOffset:
    """median / IQR per ROI over a whole dataset, as `BrainLM_Toolkit.py` computes it.

    Verified against the upstream source (retrieved 2026-08-25,
    vandijklab/BrainLM@main, toolkit/BrainLM_Toolkit.py lines 167-174 and 297-298):

        data_median_per_voxel = np.median(all_data, axis=0)
        all_data_nonzeros[all_data_nonzeros == 0] = 'nan'
        quartiles = np.nanpercentile(all_data_nonzeros, [25, 75], axis=0)
        IQR = quartiles[1,:] - quartiles[0,:]

    Zeros are excluded from the quartiles but NOT from the median, which is what upstream
    does -- `data_median_per_voxel` is taken on `all_data`, the nan-substitution happens on
    a separate copy.

    `frames_per_scan` matches upstream's `all_data[idx*200:(idx+1)*200,:] = sample[:200,:]`:
    the pool it fits on is the FIRST 200 frames of each recording, not the whole run. The
    effect on a median and an IQR is small, but the default reproduces it rather than
    quietly improving on it. Pass None to use every frame.
    """
    if frames_per_scan is not None:
        scans = [s[:, :frames_per_scan] for s in scans]
    stacked = torch.cat([s.float() for s in scans], dim=-1)          # [R, sum(T)]
    med = stacked.median(dim=-1).values
    nz = stacked.clone()
    nz[nz == 0] = float("nan")
    q1 = torch.nanquantile(nz, 0.25, dim=-1)
    q3 = torch.nanquantile(nz, 0.75, dim=-1)
    iqr = (q3 - q1).clamp_min(1e-6)
    return VoxelOffset(med / iqr, source)


def to_windows(x: torch.Tensor, window: int = PRETRAIN_WINDOW, pad_mode: str = "mean") -> torch.Tensor:
    """[R, T] -> [W, R, window], non-overlapping, padding a short scan first.

    Trailing frames beyond the last whole window are dropped, as upstream does.
    """
    r, t = x.shape
    if t < window:
        if pad_mode == "tile":
            reps = -(-window // t)
            x = x.repeat(1, reps)[:, :window]
        else:
            pad = x.mean(dim=-1, keepdim=True).repeat(1, window - t)
            x = torch.cat([x, pad], dim=-1)
        t = x.shape[-1]
    n_win = t // window
    return x[:, : n_win * window].reshape(r, n_win, window).permute(1, 0, 2).contiguous()


def prepare_scan(
    x: torch.Tensor,
    tr: float,
    amplitude: AmplitudeStats | None = None,
    offset: VoxelOffset | None = None,
    target_tr: float = PRETRAIN_TR,
    window: int = PRETRAIN_WINDOW,
    pad_mode: str = "mean",
) -> torch.Tensor:
    """One [R, T] scan -> [W, R, window] on BrainLM's pretraining distribution.

    Order matches `BrainLM_Toolkit.py`: restore amplitude, mean-centre per ROI within the
    recording, subtract the dataset constant, then resample and window.
    """
    if amplitude is not None:
        x = amplitude.apply(x)
    x = x - x.mean(dim=-1, keepdim=True)       # step 1
    if offset is not None:
        x = offset.apply(x)                    # step 2
    x = resample_linear(x, tr, target_tr)
    return to_windows(x, window=window, pad_mode=pad_mode)


def describe(tr: float, window: int = PRETRAIN_WINDOW, n_roi: int = 424) -> str:
    patches = window // PATCH
    return (
        f"TR={tr:g}s window={window} frames ({tr * window:.0f}s) "
        f"patches/ROI={patches} seq_len={n_roi * patches + 1} "
        f"token={tr * PATCH:.1f}s ({tr * PATCH / (PRETRAIN_TR * PATCH):.2f}x pretraining)"
    )
