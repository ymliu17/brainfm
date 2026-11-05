"""Brain-JEPA, through the OFFICIAL implementation in src/brainjepa_official/.

This module used to be a from-scratch reconstruction read off the checkpoint's
state_dict, because the weights shipped without code. That reconstruction recovered both
undetermined conventions correctly (token order roi_major, emb_h before the gradient half)
but still differed from the official code in three ways -- see
src/brainjepa_official/README.md. The consequence was not cosmetic: on 40 ACT scans the
two agreed at r = 0.860 per subject yet reproduced the BETWEEN-subject distance geometry
at only rho = 0.391, and a probe reads nothing but that geometry. Everything now goes
through the official code and the reconstruction is gone.

What the official path fixes, concretely:

    position embedding  the gradient half is rescaled globally to [-1, 1] before being
                        concatenated with the fixed sincos half
    input               ONE scalar mean and sd over the whole [450, T] scan, not per-ROI
    readout             mean over all 4500 tokens; the fine-tuned head adds a LayerNorm

FIXED GEOMETRY. crop_size is (450, 160): 450 ROIs and exactly 160 frames, 10 time patches
of 16. Unlike BrainLM -- whose positional encoding is sinusoidal, indexed by time position
alone, so a shorter sequence is well defined -- Brain-JEPA's position table is built for
this one grid. Callers must supply exactly 160 frames; `rct_probe.py --resample_short` and
`prepare_adni.py --short_policy adaptive` both guarantee that.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]

N_ROI = 450
SEQ_LEN = 160
PATCH_FRAMES = 16
N_TIME_PATCH = SEQ_LEN // PATCH_FRAMES      # 10
N_TOKEN = N_ROI * N_TIME_PATCH              # 4500
EMBED_DIM = 768

DEFAULT_CKPT = ROOT / "checkpoints" / "pretrained" / "brainjepa" / "jepa-ep300.pth.tar"
DEFAULT_GRAD = ROOT / "checkpoints" / "pretrained" / "brainjepa" / "gradient_mapping_450.csv"
DEFAULT_DOWNSTREAM = ROOT / "checkpoints" / "pretrained" / "brainjepa" / "downstream_example.pth"


def _official():
    """The vendored official modules.

    Imported lazily so that a tree without the checkpoints still imports this file, and
    so the heavy timm/einops chain is not paid for by callers that only want constants.
    """
    from brainjepa_official.helper import init_model
    return init_model


def _gradient(path, device):
    import pandas as pd
    g = pd.read_csv(path, header=None).values
    if g.shape != (N_ROI, 30):
        raise ValueError(f"{path}: expected ({N_ROI}, 30) brain-gradient table, got {g.shape}")
    return torch.tensor(g, dtype=torch.float32).unsqueeze(0).to(device)


def load_brainjepa(ckpt=DEFAULT_CKPT, gradient_csv=DEFAULT_GRAD, device="cpu",
                   verbose=True):
    """The pretrained encoder (jepa-ep300), built and loaded by the official code."""
    init_model = _official()
    enc, _predictor = init_model(
        device=device, patch_size=PATCH_FRAMES, crop_size=(N_ROI, SEQ_LEN),
        pred_depth=12, pred_emb_dim=384, model_name="vit_base",
        gradient_pos_embed=_gradient(gradient_csv, device),
        # flash-attn is a CUDA build not present here; 'normal' is the identical maths.
        attn_mode="normal", add_w="mapping", gradient_checkpointing=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["encoder"].items()}
    miss, unexp = enc.load_state_dict(sd, strict=False)
    if miss or unexp:
        raise RuntimeError(f"Brain-JEPA checkpoint mismatch: {len(miss)} missing, "
                           f"{len(unexp)} unexpected -- {list(miss)[:5]} {list(unexp)[:5]}")
    # The official code keeps the brain-gradient table as a plain attribute
    # (`self.gradient_pos_embed = gradient_pos_embed`), so a later `.to(cuda)` moves every
    # parameter but leaves this one on the CPU and the first forward dies inside
    # `predictor_pos_embed_proj`. Re-registering it as a buffer here makes the module
    # self-consistent under `.to()` without editing the vendored source. persistent=False
    # keeps it out of the state_dict, so the strict check above still sees 0 missing.
    g = enc.gradient_pos_embed
    del enc.gradient_pos_embed
    enc.register_buffer("gradient_pos_embed", g, persistent=False)
    if verbose:
        print(f"Brain-JEPA (official code): {len(sd)} tensors, 0 unmatched", flush=True)
    return enc.to(device).eval()


def normalise(x):
    """Whole-scan z-score, one scalar mean and sd per scan.

    Transcribed from src/datasets/hca_sex_datasets.py. NOT per-ROI: this keeps the
    between-ROI amplitude structure and only fixes the arbitrary overall gain, which is
    the opposite of what a per-ROI z-score does.
    """
    flat = x.reshape(len(x), -1)
    m = flat.mean(1)[:, None, None]
    sd = flat.std(1)[:, None, None].clamp_min(1e-6)
    return (x - m) / sd


@torch.no_grad()
def jepa_readout(model, x, readout="mean", batch=2, device=None, valid=None, layer=None):
    """x [N, 450, >=160] -> [N, D], through the official forward pass.

    mean      average of all 4500 tokens, the official `global_pool` readout    -> 768
    roi_norm  L2 norm of each ROI's mean token                                  -> 450
    roi_mean  each ROI's mean token averaged over channels                      -> 450

    `layer` selects which of the 12 transformer blocks to read, 1-indexed, with None or 12
    meaning the final output the official forward returns -- which is what every earlier
    result in this project used, because this argument did not exist. That was a real gap
    rather than a detail: BrainLM was swept over all 5 of its layers on ADNI and its best
    cell sits at layer 3, the ViTMAE releases were swept over 13 and 33 and the 650M's best
    is layer 28 of 32, and for a JEPA objective -- which predicts IN representation space --
    the last block is not where linear probes usually read best. Brain-JEPA's ADNI number of
    0.5496 was one cell out of a possible 39.

    Intermediate layers are captured with forward hooks rather than by editing
    `brainjepa_official/`, which carries only three documented local modifications and
    should keep carrying only three. The hook returns the block output BEFORE the final
    `norm`, since that norm belongs to the last block; the probe standardises its features
    anyway, so the scale difference does not matter, but the choice is stated because it is
    a choice.

    `valid` is accepted for signature compatibility with the BrainLM path and is only
    checked: Brain-JEPA has no shorter-sequence mode, so a scan that cannot supply 160
    real frames is a caller error, not something to silently truncate around.
    """
    if x.shape[1] != N_ROI:
        raise ValueError(f"Brain-JEPA needs {N_ROI} ROIs, got {x.shape[1]}")
    if x.shape[2] < SEQ_LEN:
        raise ValueError(f"Brain-JEPA needs {SEQ_LEN} frames, got {x.shape[2]}; bring the "
                         f"scan up to length first (--resample_short / --short_policy)")
    if valid is not None and int(np.asarray(valid).min()) < SEQ_LEN:
        raise ValueError(f"shortest scan has {int(np.asarray(valid).min())} valid frames, "
                         f"under the {SEQ_LEN} Brain-JEPA requires")
    dev = device or next(model.parameters()).device
    model.eval()
    blocks = getattr(model, "blocks", None)
    nb = len(blocks) if blocks is not None else None
    hook_at = None
    if layer is not None and nb is not None and 1 <= int(layer) < nb:
        hook_at = int(layer) - 1
    elif layer is not None and nb is not None and not (1 <= int(layer) <= nb):
        raise ValueError(f"layer must be in 1..{nb} (or None for the final block), "
                         f"got {layer}")
    grab = {}
    handles = []
    if hook_at is not None:
        def _hook(_m, _i, o):
            grab["t"] = o[0] if isinstance(o, tuple) else o
        handles.append(blocks[hook_at].register_forward_hook(_hook))
    out = []
    try:
      for s in range(0, len(x), batch):
        xb = normalise(x[s:s + batch, :, :SEQ_LEN].float()).unsqueeze(1).to(dev)
        t = model(xb)                                        # [B, 4500, 768]
        if hook_at is not None:
            t = grab["t"]
        if readout == "mean":
            f = t.mean(dim=1)
        elif readout in ("roi_norm", "roi_mean"):
            r = t.view(t.shape[0], N_ROI, N_TIME_PATCH, EMBED_DIM).mean(dim=2)
            f = r.norm(dim=2) if readout == "roi_norm" else r.mean(dim=2)
        else:
            raise ValueError(f"unknown readout {readout!r}")
        out.append(f.float().cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(out).numpy()

@torch.no_grad()
def jepa_readout_multi(model, x, readouts=("mean", "roi_norm", "roi_mean"),
                       layers=None, batch=2, device=None, valid=None):
    """{(layer, readout): [N, D]} for every requested block, in ONE forward pass.

    Twelve separate passes would be the obvious implementation and it is not affordable:
    a ViT-B over 4500 tokens runs ADNI's 994 scans in roughly 25 minutes, so sweeping 12
    layers serially is five hours. The hooks here reduce each block's output to its readouts
    INSIDE the hook, so only 12 x [B, 768] and 12 x [B, 450] are ever held rather than
    12 x [B, 4500, 768] -- 331 MB per batch of 2 at float32.

    `layers` is 1-indexed over the blocks; the last entry is read from the model's own
    return value so it stays bit-identical to `jepa_readout(..., layer=None)`, which is
    what every published Brain-JEPA number used.
    """
    if x.shape[1] != N_ROI:
        raise ValueError(f"Brain-JEPA needs {N_ROI} ROIs, got {x.shape[1]}")
    if x.shape[2] < SEQ_LEN:
        raise ValueError(f"Brain-JEPA needs {SEQ_LEN} frames, got {x.shape[2]}")
    if valid is not None and int(np.asarray(valid).min()) < SEQ_LEN:
        raise ValueError(f"shortest scan has {int(np.asarray(valid).min())} valid frames")
    dev = device or next(model.parameters()).device
    model.eval()
    nb = len(model.blocks)
    layers = list(range(1, nb + 1)) if layers is None else sorted(set(int(v) for v in layers))
    if any(v < 1 or v > nb for v in layers):
        raise ValueError(f"layers must be within 1..{nb}, got {layers}")

    def reduce(t):
        o = {}
        if "mean" in readouts:
            o["mean"] = t.mean(dim=1).float().cpu()
        if "roi_norm" in readouts or "roi_mean" in readouts:
            r = t.view(t.shape[0], N_ROI, N_TIME_PATCH, t.shape[-1]).mean(dim=2)
            if "roi_norm" in readouts:
                o["roi_norm"] = r.norm(dim=2).float().cpu()
            if "roi_mean" in readouts:
                o["roi_mean"] = r.mean(dim=2).float().cpu()
        return o

    acc: dict = {}
    handles = []
    for li in layers:
        if li == nb:
            continue                       # taken from the model's own return value
        def _hook(_m, _i, o, li=li):
            t = o[0] if isinstance(o, tuple) else o
            for k, v in reduce(t).items():
                acc.setdefault((li, k), []).append(v)
        handles.append(model.blocks[li - 1].register_forward_hook(_hook))
    try:
        for s in range(0, len(x), batch):
            xb = normalise(x[s:s + batch, :, :SEQ_LEN].float()).unsqueeze(1).to(dev)
            t = model(xb)
            if nb in layers:
                for k, v in reduce(t).items():
                    acc.setdefault((nb, k), []).append(v)
            print(f"  jepa {min(s + batch, len(x))}/{len(x)}", end="\r", flush=True)
    finally:
        for h in handles:
            h.remove()
    print()
    return {k: torch.cat(v).numpy() for k, v in acc.items()}
