"""fMRI-LM stage-1 VQ tokenizer, loaded through the AUTHORS' code.

`third_party/fMRI-LM` is the official repo (yuxiangwei0808/fMRI-LM, CVPR 2026, arXiv
2511.21760). Nothing here reimplements the model: `VQ` and its ViT come from that clone, so
the architecture, the quantiser's distance metric and the encode path are the authors'.

Two deviations, both forced and both recorded because they bound what can be concluded.

1. INPUT NORMALISATION IS NOT THE AUTHORS'. `dataset.fMRIDataSet` applies a per-ROI robust
   z-score, `(X - median) / iqr`, with median and iqr read from a `normalization_params.npz`
   that ships with THEIR preprocessed corpus. That file is not in the repo, not in the
   released checkpoint directory, and not derivable from either. `robust_norm_params` below
   therefore computes the same statistics on whichever corpus is being fed. The OPERATION is
   the authors'; the CONSTANTS are ours. Anything measured here is conditional on that, and
   `--norm none` reproduces the un-normalised input so the two can be compared.
2. `conf.attn_mode` is `flash_attn` and flash-attn is not installed. In the authors'
   `Attention.forward` the flash branch is guarded by `HAS_FLASH_ATTENTION`, and with the
   flag false and the mode still `flash_attn` NEITHER branch runs. `attn_mode` is therefore
   forced to `normal`, which is the same attention computed densely.

ADNI: do not evaluate this encoder on it. arXiv 2511.21760 §4.1 states stages 1-2 use UKB and
ABCD only, but the checkpoint's `conf` does not record `--dataset_dir`, so which corpus
produced THIS file cannot be verified; and ADNI with target `AD` is in the authors' own
downstream suite either way, which is not the position BrainLM and Brain-JEPA are in. The
leave-one-study-out design (`rct_loso.py`) needs no ADNI at all and is the clean comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "fMRI-LM"
FMRILM_CKPT = ROOT / "checkpoints" / "pretrained" / "fmri-lm" / "ckpt-best.pt"

NUM_ROIS, NUM_FRAMES = 450, 160


def _official():
    """Import the authors' VQ. Kept lazy so the repo is only needed when this is used."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from quantizers.vq import VQ
    return VQ


def load_fmrilm(ckpt=FMRILM_CKPT, device=None, verbose=True, random_init=False):
    """The authors' VQ with the released stage-1 weights. Strict on the VQ subtree.

    `random_init=True` builds the same architecture from the same config and skips the load.
    It is the control that makes a probe result attributable, and it is needed here rather
    than optional: with these weights the reconstruction is uncorrelated with its own input
    (scan-level r = +0.0001 under three different input normalisations, against MSE 2.070
    where predicting zero scores 2.035), and a random-init model of this architecture spreads
    over 3485 codebook entries where the loaded weights use 14. So the loaded file behaves
    like an untrained one on the decode path, and only a side-by-side probe can say whether
    the ENCODER is in the same state.
    """
    from omegaconf import OmegaConf
    blob = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    conf = OmegaConf.create(OmegaConf.to_container(blob["conf"], resolve=True))
    conf.attn_mode = "normal"                      # see the module docstring
    conf.in_chans = 1                              # encoder side; VQ rewrites it for the decoder
    conf.setdefault("img_size", [NUM_ROIS, NUM_FRAMES])
    VQ = _official()
    m = VQ(conf)
    if random_init:
        if verbose:
            print("fMRI-LM stage-1 VQ: RANDOM INIT control, weights NOT loaded", flush=True)
    else:
        sub = {k[len("VQ."):]: v for k, v in blob["model"].items() if k.startswith("VQ.")}
        m.load_state_dict(sub, strict=True)
        if verbose:
            n = sum(p.numel() for p in m.parameters()) / 1e6
            print(f"fMRI-LM stage-1 VQ: {n:.1f}M params, {len(sub)} tensors, "
                  f"strict load OK", flush=True)
    m.eval()
    return m.to(device) if device is not None else m


def robust_norm_params(x, eps=1e-6):
    """Per-ROI median and IQR over a corpus. x: [N, R, T] -> (median[R,1], iqr[R,1]).

    The authors' `norm='robust'` statistic, computed on the corpus at hand because theirs is
    not distributed. Pooling over scans AND time matches `normalization_params.npz`, which
    holds one number per ROI rather than one per scan.
    """
    v = x.permute(1, 0, 2).reshape(x.shape[1], -1).float()
    q1, med, q3 = (v.quantile(q, dim=1) for q in (0.25, 0.5, 0.75))
    return med[:, None], (q3 - q1).clamp_min(eps)[:, None]


@torch.no_grad()
def fmrilm_readout(model, x, readout="mean", layer=None, batch=8, device=None, norm=None):
    """[N, d] for one (layer, readout). `norm` is (median, iqr) from `robust_norm_params`."""
    out = fmrilm_readout_multi(model, x, (readout,), [layer] if layer else None,
                               batch, device, norm)
    lay = layer if layer else len(model.encoder.blocks)
    return out[(lay, readout)]


@torch.no_grad()
def fmrilm_readout_multi(model, x, readouts=("mean", "roi_norm", "roi_var", "emb_fc"),
                         layers=None, batch=8, device=None, norm=None):
    """{(layer, readout): [N, d]} in ONE forward pass per batch.

    Hooks reduce inside the hook, as in `brainjepa.jepa_readout_multi`: keeping 12 layers of
    [B, 450, 768] costs 12x the activation memory for nothing, since every readout is a
    reduction over the token axis.
    """
    enc = model.encoder
    dev = device or next(model.parameters()).device
    nb = len(enc.blocks)
    layers = list(range(1, nb + 1)) if layers is None else [l for l in layers if l >= 1]
    cache, out = {}, {}

    def reduce_to(h, tag):
        r = h.float()
        if "mean" in readouts:
            cache[(tag, "mean")] = r.mean(dim=1).cpu()
        if "roi_norm" in readouts:
            cache[(tag, "roi_norm")] = r.norm(dim=2).cpu()
        if "roi_var" in readouts:
            cache[(tag, "roi_var")] = r.var(dim=2).cpu()
        if "emb_fc" in readouts:
            # FC over the learned parcellation, the same operator used for BrainLM's emb_fc,
            # so the two families' emb_fc columns mean the same thing.
            z = r - r.mean(dim=2, keepdim=True)
            z = z / z.norm(dim=2, keepdim=True).clamp_min(1e-8)
            c = z @ z.transpose(1, 2)
            iu = torch.triu_indices(c.shape[1], c.shape[2], offset=1, device=c.device)
            cache[(tag, "emb_fc")] = c[:, iu[0], iu[1]].cpu()

    hooks = [enc.blocks[i - 1].register_forward_hook(
        lambda _m, _i, o, t=i: reduce_to(o[0] if isinstance(o, tuple) else o, t))
        for i in layers]
    try:
        for s in range(0, len(x), batch):
            cache.clear()
            xb = x[s:s + batch].to(dev).float()
            if xb.shape[2] > NUM_FRAMES:
                xb = xb[:, :, :NUM_FRAMES]
            if norm is not None:
                med, iqr = norm
                xb = (xb - med.to(dev)) / iqr.to(dev)
            enc(xb)
            for k, v in cache.items():
                out.setdefault(k, []).append(v)
    finally:
        for h in hooks:
            h.remove()
    return {k: torch.cat(v).numpy() for k, v in out.items()}


@torch.no_grad()
def fmrilm_codes(model, x, batch=8, device=None, norm=None):
    """Discrete token ids [N, 450] through the authors' quantiser."""
    dev = device or next(model.parameters()).device
    ids = []
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(dev).float()[:, :, :NUM_FRAMES]
        if norm is not None:
            med, iqr = norm
            xb = (xb - med.to(dev)) / iqr.to(dev)
        ids.append(model.get_codebook_indices(xb).cpu())
    return torch.cat(ids)
