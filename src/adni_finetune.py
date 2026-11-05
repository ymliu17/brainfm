"""ADNI domain-adaptive fine-tuning model (stage 1).

RECONSTRUCTION NOTICE — rebuilt from the manuscript's "Model fine-tuning" section.
The original ADNI fine-tuning script is not part of this handover package.

Manuscript:
  "supervised fine-tuning was performed on the ADNI dataset ... formulated as a
   binary classification task distinguishing participants diagnosed with AD vs. HC.
   ... lightweight Low-Rank Adaptation (LoRA) modules were inserted into the query,
   key, value, and output projection layers of the self-attention blocks. Most
   pretrained FM weights remained frozen, while the final transformer block was
   unfrozen to allow limited adaptation of higher-level representations. ... the
   downstream classifier consisted of a two-layer MLP operating on the CLS token
   embedding produced by the BrainLM. Model optimization was conducted using the
   AdamW optimizer with differential learning rates applied to the adapted rsfMRI FM
   weights and the downstream classifier. A cosine learning rate scheduler with
   warmup was used ... Early stopping was applied based on validation loss ... the
   adapted BrainLM model (denoted as BrainLM-ADNI) was used as a frozen feature
   extractor for all downstream ... No further weight updates were performed during
   linear probing."

The state-dict is saved under `model_state_dict` with `brainlm.base_model.model.*`
key prefixes so that src/model_loader.py can consume it directly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

import config as C
from classifiers import READOUT_WIDTH
from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining


def _build_backbone_config() -> BrainLMConfig:
    # Fine-tuning sees full sequences; masking is disabled (mask_ratio=0) so the CLS
    # embedding is computed from all patches, matching downstream feature extraction.
    return BrainLMConfig(
        hidden_size=C.HIDDEN_SIZE,
        timepoint_patching_size=C.TIMEPOINT_PATCHING_SIZE,
        num_hidden_layers=C.NUM_HIDDEN_LAYERS,
        num_attention_heads=C.NUM_ATTENTION_HEADS,
        intermediate_size=C.INTERMEDIATE_SIZE,
        num_brain_voxels=C.NUM_BRAIN_VOXELS,
        num_timepoints_per_voxel=C.NUM_TIMEPOINTS_PER_VOXEL,
        decoder_hidden_size=C.DECODER_HIDDEN_SIZE,
        decoder_num_hidden_layers=C.DECODER_NUM_HIDDEN_LAYERS,
        decoder_num_attention_heads=C.DECODER_NUM_ATTENTION_HEADS,
        decoder_intermediate_size=C.DECODER_INTERMEDIATE_SIZE,
        mask_ratio=C.MASK_RATIO,
    )


class BrainLMADNIClassifier(nn.Module):
    """LoRA-adapted BrainLM + unfrozen final encoder block + 2-layer MLP head.

    Attribute name `brainlm` (the PEFT model) makes the saved state-dict keys line
    up with what src/model_loader.py expects.
    """

    def __init__(self, pretrained_backbone_path: str | None = None,
                 lora_r: int | None = None, unfreeze_last_block: bool = True):
        super().__init__()
        base = BrainLMForPretraining(_build_backbone_config())
        if pretrained_backbone_path:
            self._load_pretrained_backbone(base, pretrained_backbone_path)

        # LoRA rank and whether the final encoder block trains are the two knobs that
        # set trainable capacity. At the default r=32 with the last block unfrozen this
        # model has 1.7M trainable parameters for ~365 training subjects; alpha is scaled
        # with r so the effective LoRA gain stays fixed as rank changes.
        lora_r = C.LORA_R if lora_r is None else lora_r
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=int(C.LORA_ALPHA * lora_r / C.LORA_R),
            lora_dropout=C.LORA_DROPOUT,
            target_modules=list(C.LORA_TARGET_MODULES),
        )
        self.brainlm = get_peft_model(base, lora_config)

        # Two-layer MLP head on the CLS token.
        self.classifier = nn.Sequential(
            nn.Linear(C.HIDDEN_SIZE, C.CLASSIFIER_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(C.CLASSIFIER_EMBED_DIM, 1),
        )

        self._set_trainable(unfreeze_last_block=unfreeze_last_block)

    @staticmethod
    def _load_pretrained_backbone(base, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model_sd = base.state_dict()
        filtered = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        base.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} pretrained backbone tensors from {path}", flush=True)

    def _set_trainable(self, unfreeze_last_block: bool = True):
        """Freeze everything except LoRA params, optionally the final encoder block, and head."""
        vit = self.brainlm.base_model.model.vit
        n_layers = len(vit.encoder.layer)
        for name, p in self.brainlm.named_parameters():
            is_lora = "lora_" in name.lower()
            is_final_block = unfreeze_last_block and f"encoder.layer.{n_layers - 1}." in name
            p.requires_grad = bool(is_lora or is_final_block)
        for p in self.classifier.parameters():
            p.requires_grad = True

    def encode_cls(self, signal_vectors, xyz_vectors):
        out = self.brainlm.base_model.model.vit(
            signal_vectors=signal_vectors,
            xyz_vectors=xyz_vectors,
            output_hidden_states=True,
            return_dict=True,
            noise=None,
        )
        return out.hidden_states[-1][:, 0, :]

    def forward(self, signal_vectors, xyz_vectors):
        cls = self.encode_cls(signal_vectors, xyz_vectors)
        return self.classifier(cls).squeeze(-1)

    def param_groups(self, fm_lr: float, head_lr: float):
        """Differential learning rates: adapted FM weights vs classifier head."""
        fm = [p for n, p in self.brainlm.named_parameters() if p.requires_grad]
        head = [p for p in self.classifier.parameters() if p.requires_grad]
        return [
            {"params": fm, "lr": fm_lr},
            {"params": head, "lr": head_lr},
        ]


class BrainLMPairedRegressor(BrainLMADNIClassifier):
    """Same LoRA-adapted backbone, but scoring a PAIR of scans against a change score.

    The intermediate ADNI task is deliberately built to match the RCT target task in both
    input structure and label semantics: two scans of one subject in, change in a cognitive
    composite out. This class therefore encodes T1 and T2 with the SAME weights and
    combines them the way `finetune_decline.py` does downstream, so the architecture the
    RCT head sits on is the architecture ADNI trained.

    `combine`. Note that {diff, mean} is an orthogonal reparameterisation of {h(T1), h(T2)},
    so running both separates *which component* carries the signal in a way `concat` cannot:

        diff    h(T2) - h(T1)         -- the change in representation; the natural match for
                                         a change target, and invariant to whatever part of
                                         h is constant within a subject
        mean    (h(T1) + h(T2)) / 2   -- the subject's average state. Averaging two scans
                                         halves the measurement noise, so this is the most
                                         RELIABLE summary of the subject available, and it
                                         is what carries baseline level -- who has room to
                                         improve, and regression toward the mean. Written as
                                         a mean rather than a sum so it shares a scale with
                                         the other options; a linear head absorbs the
                                         factor 2 either way.
        concat  [h(T1), h(T2)]        -- both, at twice the head width. Spans the same space
                                         as diff+mean but spends 2x the head parameters and
                                         cannot tell you which half mattered.
        t1      h(T1) only            -- baseline scan only. The deployable setting: predict
                                         who will change from a single scan taken before the
                                         intervention. Also the strictest test, since it
                                         cannot see the outcome window at all.
        t2      h(T2) only            -- ablation. If this matches `diff`, the model is
                                         reading end state rather than change.

    `readout` defaults to `mean_last2` rather than the `cls` used by the diagnosis task.
    BrainLM is a masked autoencoder, so nothing in pretraining asks the CLS token to
    summarise the sequence; measured on ADNI covariates the patch-token mean over the last
    two layers beats CLS by ~0.06 AUC. Keeping `cls` available makes that comparable.
    """

    COMBINE = {"diff": 1, "mean": 1, "concat": 2, "t1": 1, "t2": 1}

    def __init__(self, pretrained_backbone_path: str | None = None,
                 lora_r: int | None = None, unfreeze_last_block: bool = True,
                 readout: str = "mean_last2", combine: str = "diff"):
        if combine not in self.COMBINE:
            raise ValueError(f"combine must be one of {sorted(self.COMBINE)}, got {combine!r}")
        if readout not in READOUT_WIDTH:
            raise ValueError(f"readout must be one of {sorted(READOUT_WIDTH)}, got {readout!r}")
        super().__init__(pretrained_backbone_path=pretrained_backbone_path, lora_r=lora_r,
                         unfreeze_last_block=unfreeze_last_block)
        self.readout = readout
        self.combine = combine
        in_dim = C.HIDDEN_SIZE * READOUT_WIDTH[readout] * self.COMBINE[combine]
        # Replaces the inherited binary head. Same two-layer shape so the d=128 embedding
        # the PLS analysis reads stays available at `classifier[:2]`.
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, C.CLASSIFIER_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(C.CLASSIFIER_EMBED_DIM, 1),
        )
        for p in self.classifier.parameters():
            p.requires_grad = True

    def encode_one(self, signal_vectors, xyz_vectors):
        """Pool one scan's tokens with `self.readout`. [B, 424, T] -> [B, hidden*width]."""
        out = self.brainlm.base_model.model.vit(
            signal_vectors=signal_vectors, xyz_vectors=xyz_vectors,
            output_hidden_states=True, return_dict=True, noise=None,
        )
        hs = out.hidden_states
        if self.readout == "cls":
            return hs[-1][:, 0, :]
        if self.readout == "mean":
            return hs[-1][:, 1:, :].mean(dim=1)
        if self.readout == "cls+mean":
            return torch.cat([hs[-1][:, 0, :], hs[-1][:, 1:, :].mean(dim=1)], dim=1)
        if self.readout == "last2":
            return torch.cat([hs[-1][:, 0, :], hs[-2][:, 0, :]], dim=1)
        return torch.cat([hs[-1][:, 1:, :].mean(dim=1), hs[-2][:, 1:, :].mean(dim=1)], dim=1)

    def embed(self, signal_pair, xyz_vectors):
        """[B, 424, T, 2] -> the combined representation the head sees."""
        if signal_pair.dim() != 4 or signal_pair.shape[-1] != 2:
            raise ValueError(f"expected [B, 424, T, 2], got {tuple(signal_pair.shape)}")
        # `t1` needs only the first scan; skipping the second halves the compute and,
        # more importantly, guarantees the follow-up scan cannot leak into it.
        h1 = self.encode_one(signal_pair[..., 0], xyz_vectors)
        if self.combine == "t1":
            return h1
        h2 = self.encode_one(signal_pair[..., 1], xyz_vectors)
        if self.combine == "diff":
            return h2 - h1
        if self.combine == "mean":
            return 0.5 * (h1 + h2)
        if self.combine == "t2":
            return h2
        return torch.cat([h1, h2], dim=1)

    def forward(self, signal_pair, xyz_vectors):
        return self.classifier(self.embed(signal_pair, xyz_vectors)).squeeze(-1)
