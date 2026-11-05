"""Input path for the BrainLM 111M / 650M releases.

Those two are `ViTMAEForPreTraining`, not `BrainLMForPretraining`: they have 2D patch
embeddings and no signal or xyz projection, so they do not take a time series the way the
13M does. The upstream pipeline reaches them by treating a recording as an image.

Transcribed from `brainlm_tutorial.ipynb` in https://github.com/vandijklab/BrainLM:

  1. reorder ROIs by their **Y** coordinate -- neighbouring ROIs get convolved into the
     same patch here, so their order is load-bearing in a way it is not for the 13M
  2. divide by the dataset constant `max_val_to_scale = 5.6430855`
  3. repeat to 3 channels
  4. pad symmetrically to the model's image size with the constant **-1**
     (424x200 -> 432x432 for the 111M, 434x434 for the 650M)
  5. take `vit(...).last_hidden_state[:, 0, :]` as the CLS embedding

Two details the tutorial leaves implicit and that matter for frozen-feature evaluation:

* **Masking has to be switched off explicitly, and so does the shuffle.**
  `ViTMAEEmbeddings.forward` always calls `random_masking`. At the config default
  `mask_ratio=0.75` the encoder sees a random quarter of the patches and the CLS token
  differs on every call. Setting `mask_ratio=0` keeps every patch, but the argsort still
  *permutes* them: position embeddings are added first so the result is mathematically
  unchanged, yet the reduction order in attention changes and the features move by ~3e-6
  between calls. Passing an all-zero `noise` with `train_mode != "auto_encode"` makes the
  argsort the identity and `len_keep` the full sequence, which is both exact and reusable
  from a cache. `assert_deterministic` verifies it rather than trusting the reasoning.
* **flash attention is unnecessary here.** The upstream notebook installs it, but the
  sequences are 730 (111M) and 962 (650M) tokens, which is small; the 13M's 2969-6361
  token sequences are the ones that needed it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

#: Upstream constant, documented there as dataset-specific.
MAX_VAL_TO_SCALE = 5.6430855

#: Constant the recording is padded with, out to the model's square image size.
PAD_VALUE = -1.0


def load_vitmae(model_dir: str | Path, device=None):
    """Load a BrainLM ViTMAE release as a frozen, deterministic encoder."""
    from transformers import ViTMAEConfig

    from brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining

    model_dir = Path(model_dir)
    config = ViTMAEConfig.from_pretrained(model_dir)
    config.mask_ratio = 0.0          # keep every patch; see the module docstring
    config.train_mode = "inference"  # so `noise` is honoured instead of resampled
    model = ViTMAEForPreTraining.from_pretrained(model_dir, config=config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if device is not None:
        model.to(device)
    return model, config


def prepare_batch(x: torch.Tensor, roi_order: np.ndarray, image_size) -> torch.Tensor:
    """[B, R, T] recordings -> [B, 3, H, W] padded images, following steps 1-4."""
    if x.dim() != 3:
        raise ValueError(f"expected [B, R, T], got {tuple(x.shape)}")
    h, w = (image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size))
    if x.shape[1] > h or x.shape[2] > w:
        raise ValueError(
            f"recording {tuple(x.shape[1:])} does not fit the model's {h}x{w} image"
        )
    x = x[:, roi_order, :] / MAX_VAL_TO_SCALE
    x = x.unsqueeze(1).repeat(1, 3, 1, 1)
    ph, pw = h - x.shape[2], w - x.shape[3]
    # Symmetric padding, as upstream does. Odd totals put the extra row/column last.
    return F.pad(x, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), "constant", PAD_VALUE)


@torch.no_grad()
def encode(model, pixel_values: torch.Tensor, readout: str = "cls") -> torch.Tensor:
    """CLS token (or mean over patch tokens) from the frozen encoder.

    The all-zero `noise` keeps every patch and leaves the token order untouched, so the
    embedding is bit-reproducible across calls and across processes.
    """
    n_patches = model.vit.embeddings.patch_embeddings.num_patches
    noise = torch.zeros(pixel_values.shape[0], n_patches, device=pixel_values.device)
    out = model.vit(pixel_values=pixel_values, noise=noise).last_hidden_state
    if readout == "cls":
        return out[:, 0, :]
    if readout == "mean":
        return out[:, 1:, :].mean(dim=1)
    if readout == "cls+mean":
        return torch.cat([out[:, 0, :], out[:, 1:, :].mean(dim=1)], dim=-1)
    raise ValueError(f"unknown readout {readout!r}")


@torch.no_grad()
def assert_deterministic(model, pixel_values: torch.Tensor) -> None:
    """Two identical forwards must agree, or `mask_ratio` is still masking."""
    a = encode(model, pixel_values)
    b = encode(model, pixel_values)
    if not torch.equal(a, b):
        raise RuntimeError(
            f"ViTMAE encoder is not deterministic (max diff {(a - b).abs().max():.3e}). "
            "Either mask_ratio is not 0 (a random subset of patches is being kept) or the "
            "token order is still being shuffled."
        )
