"""Frozen-backbone classifiers for concat / t1-only / diff modes.

Two properties this module guarantees, both of which the original version broke:

1. The backbone stays in eval mode. `nn.Module.train()` recurses into every child,
   so calling `model.train()` on the wrapper used to re-enable dropout (positional
   encoding 0.1, attention 0.1, LoRA 0.1) inside the "frozen" feature extractor,
   giving training and validation systematically different features. `train()` is
   overridden below to leave the backbone alone.

2. Checkpoints contain the head only. The backbone is frozen, so storing it is pure
   duplication -- and because concat/diff bind the same module to two attributes,
   `state_dict()` would write it out twice. `save_head` / `load_head` handle the head;
   the backbone is rebuilt from `checkpoints/pretrained/` on load.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import config as C


def _encoder(model):
    """Return the BrainLM encoder (vit), whether or not the model is PEFT-wrapped.

    Running the encoder directly extracts the embedding of the frozen FM without
    invoking the pretraining decoder / reconstruction loss. A LoRA checkpoint arrives
    wrapped (model.base_model = LoraModel; .model = BrainLMForPretraining; .vit), while a
    plain pretrained checkpoint arrives as BrainLMForPretraining itself.
    """
    for path in (("base_model", "model", "vit"), ("vit",)):
        obj = model
        try:
            for attr in path:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"cannot locate the encoder on {type(model).__name__}")


#: How the encoder's token sequence is reduced to one vector per scan.
#:
#: The default `cls` is what the handover code used. It is worth questioning: a ridge
#: logistic regression on raw FC edges reaches AUC 0.772 on ADNI stage-1 where the CLS
#: readout reaches 0.531 (the original cohort audit 4b-8), so the pooling may be discarding
#: information the encoder actually holds. BrainLM's pretraining objective is patch
#: reconstruction, not a CLS-token task, which gives no particular reason for the CLS
#: token to be a good summary.
#:
#:   cls        CLS token of the last layer                      [hidden]
#:   mean       mean over patch tokens of the last layer         [hidden]
#:   cls+mean   both concatenated                                [2*hidden]
#:   last2      CLS of the last two layers concatenated          [2*hidden]
#:   mean_last2 patch-token mean of the last two layers          [2*hidden]
READOUTS = ("cls", "mean", "cls+mean", "last2", "mean_last2")

#: Multiplier on `hidden_size` for each readout.
READOUT_WIDTH = {"cls": 1, "mean": 1, "cls+mean": 2, "last2": 2, "mean_last2": 2}


def get_cls_feature(model, x, xyz, readout: str = "cls"):
    """Pool the frozen backbone's tokens into one vector per scan.

    Deterministic: the encoder is in eval mode, so no patch masking and no dropout are
    applied. Token index 0 is the CLS token; the rest are (voxel, time-patch) tokens.
    """
    if readout not in READOUTS:
        raise ValueError(f"unknown readout {readout!r}; expected one of {READOUTS}")
    xyz_batch = xyz.to(x.device)
    if xyz_batch.dim() == 2:  # [num_voxels, 3] -> [batch, num_voxels, 3]
        xyz_batch = xyz_batch.unsqueeze(0).repeat(x.shape[0], 1, 1)
    with torch.no_grad():
        out = _encoder(model)(
            signal_vectors=x,
            xyz_vectors=xyz_batch,
            noise=None,
            return_dict=True,
            output_hidden_states=True,
        )
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            last = out.last_hidden_state
            hs = (last, last)
        if readout == "cls":
            return hs[-1][:, 0, :]
        if readout == "mean":
            return hs[-1][:, 1:, :].mean(dim=1)
        if readout == "cls+mean":
            return torch.cat([hs[-1][:, 0, :], hs[-1][:, 1:, :].mean(dim=1)], dim=1)
        if readout == "last2":
            return torch.cat([hs[-1][:, 0, :], hs[-2][:, 0, :]], dim=1)
        return torch.cat([hs[-1][:, 1:, :].mean(dim=1), hs[-2][:, 1:, :].mean(dim=1)], dim=1)


def build_head(in_dim: int) -> nn.Sequential:
    """Two-layer MLP head. `head[:2]` is the d=128 embedding used by the PLS analysis."""
    return nn.Sequential(
        nn.Linear(in_dim, C.CLASSIFIER_EMBED_DIM),
        nn.ReLU(),
        nn.Linear(C.CLASSIFIER_EMBED_DIM, 1),
    )


class _FrozenBackboneClassifier(nn.Module):
    """Base class that keeps the backbone frozen *and* in eval mode."""

    #: attribute names holding the (shared) backbone module
    backbone_attrs: tuple[str, ...] = ()

    def train(self, mode: bool = True):
        """Set only the head's training mode; the backbone stays in eval."""
        self.training = mode
        self.classifier.train(mode)
        for attr in self.backbone_attrs:
            getattr(self, attr).eval()
        return self

    def head_state_dict(self):
        return self.classifier.state_dict()

    def load_head_state_dict(self, sd):
        return self.classifier.load_state_dict(sd)


class ConcatClassifier(_FrozenBackboneClassifier):
    """Concatenate T1 and T2 CLS features -> MLP."""

    backbone_attrs = ("brainlm_t1", "brainlm_t2")

    def __init__(self, brainlm):
        super().__init__()
        self.brainlm_t1 = brainlm
        self.brainlm_t2 = brainlm  # shared weights (same module object)
        self.classifier = build_head(brainlm.config.hidden_size * 2)
        self.train(False)

    def features(self, t1, t2, xyz):
        return torch.cat(
            [get_cls_feature(self.brainlm_t1, t1, xyz), get_cls_feature(self.brainlm_t2, t2, xyz)],
            dim=1,
        )

    def forward(self, t1, t2, xyz):
        return self.classifier(self.features(t1, t2, xyz)).squeeze(-1)


class T1OnlyClassifier(_FrozenBackboneClassifier):
    """Use only T1 CLS feature -> MLP."""

    backbone_attrs = ("brainlm",)

    def __init__(self, brainlm):
        super().__init__()
        self.brainlm = brainlm
        self.classifier = build_head(brainlm.config.hidden_size)
        self.train(False)

    def features(self, t1, t2, xyz):
        return get_cls_feature(self.brainlm, t1, xyz)

    def forward(self, t1, t2, xyz):
        return self.classifier(self.features(t1, t2, xyz)).squeeze(-1)


class DiffClassifier(_FrozenBackboneClassifier):
    """Use (T2 - T1) CLS feature difference -> MLP."""

    backbone_attrs = ("brainlm_t1", "brainlm_t2")

    def __init__(self, brainlm):
        super().__init__()
        self.brainlm_t1 = brainlm
        self.brainlm_t2 = brainlm  # shared weights
        self.classifier = build_head(brainlm.config.hidden_size)
        self.train(False)

    def features(self, t1, t2, xyz):
        return get_cls_feature(self.brainlm_t2, t2, xyz) - get_cls_feature(self.brainlm_t1, t1, xyz)

    def forward(self, t1, t2, xyz):
        return self.classifier(self.features(t1, t2, xyz)).squeeze(-1)


_MODES = {"concat": ConcatClassifier, "t1": T1OnlyClassifier, "diff": DiffClassifier}


def build_classifier(mode: str, brainlm) -> nn.Module:
    if mode not in _MODES:
        raise ValueError(f"Unknown mode: {mode}. Expected {'|'.join(_MODES)}.")
    return _MODES[mode](brainlm)


def head_input_dim(mode: str, hidden_size: int, readout: str = "cls") -> int:
    """Width of the feature vector the head consumes, per mode and readout."""
    per_scan = hidden_size * READOUT_WIDTH[readout]
    return per_scan * 2 if mode == "concat" else per_scan


def combine_features(mode: str, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Build the head input from cached per-timepoint CLS features."""
    if mode == "concat":
        return torch.cat([f1, f2], dim=1)
    if mode == "t1":
        return f1
    if mode == "diff":
        return f2 - f1
    raise ValueError(f"Unknown mode: {mode}")


def save_head(head: nn.Module, path: str | Path, meta: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"head_state_dict": head.state_dict(), "meta": meta or {}}, path)


def load_head(path: str | Path, in_dim: int) -> nn.Sequential:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    head = build_head(in_dim)
    head.load_state_dict(obj["head_state_dict"] if "head_state_dict" in obj else obj)
    return head
