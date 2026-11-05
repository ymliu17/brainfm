"""Load frozen LoRA BrainLM backbone used by mainline finetuning."""
from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model

from brainlm_mae.configuration_brainlm import BrainLMConfig
from brainlm_mae.modeling_brainlm import BrainLMForPretraining

import config as C


def load_brainlm_forpretraining(pretrained_path: str | Path):
    """Create LoRA BrainLM and load DX-binary LoRA checkpoint; freeze all params.

    Architecture and masking come from src/config.py so the code stays aligned
    with the manuscript. The returned module is frozen and left in eval mode;
    classifiers.py keeps it there even when the wrapper is switched to train().
    """
    pretrained_path = Path(pretrained_path)

    config = BrainLMConfig(
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

    # Read the checkpoint before building anything: `get_peft_model` rewrites the target
    # modules of `base_model` in place (query.weight -> query.base_layer.weight), so a
    # plain state dict must be loaded into an UNWRAPPED model or 72 of its 124 tensors
    # land as "unexpected" and are silently dropped.
    print(f"Loading checkpoint: {pretrained_path}", flush=True)
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    base_model = BrainLMForPretraining(config)

    # Unwrap a training-checkpoint envelope if there is one. Dispatch must NOT depend on
    # this: `scripts/dapt_adni.py` writes plain BrainLM weights under "model_state_dict",
    # and an earlier version of this function keyed off that name, wrapped the model in
    # PEFT, matched none of the 124 tensors, and printed "Loaded LoRA checkpoint
    # successfully" over a randomly initialised model.
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError(f"{pretrained_path.name} does not contain a state dict")

    # Decide from the key names, which is the only reliable signal.
    is_lora = any("lora_" in k or ".base_model.model." in k for k in state_dict)

    def _report(missing, unexpected, kind):
        loaded = len(state_dict) - len(unexpected)
        if loaded == 0:
            raise ValueError(
                f"{pretrained_path.name}: none of its {len(state_dict)} tensors matched "
                f"the {kind} model. Check the architecture in src/config.py. Refusing to "
                f"return a randomly initialised model."
            )
        if len(unexpected) > 0.1 * len(state_dict):
            raise ValueError(
                f"{pretrained_path.name}: {len(unexpected)} of {len(state_dict)} tensors "
                f"did not match the {kind} model. That is too many to be a benign "
                f"difference; the checkpoint and the architecture disagree."
            )
        print(
            f"Loaded {kind} checkpoint: {loaded}/{len(state_dict)} tensors "
            f"(missing {len(missing)}, unexpected {len(unexpected)})",
            flush=True,
        )

    if not is_lora:
        _report(*base_model.load_state_dict(state_dict, strict=False), "plain")
        for p_ in base_model.parameters():
            p_.requires_grad = False
        base_model.eval()
        return base_model

    # The rank must come from the CHECKPOINT, not from src/config.py. Adapters trained at a
    # different rank than the configured default produce shape-mismatched tensors, and
    # `load_state_dict(strict=False)` does NOT rescue that -- it either raises or drops them.
    # Measured: every scripts/contrastive_adni.py checkpoint was written at r=8 while
    # C.LORA_R is 32, so none of them was loadable by any downstream probe. That is the
    # reason the contrastive arm never produced a downstream number.
    rank = C.LORA_R
    for k, v in state_dict.items():
        if "lora_A" in k and v.ndim == 2:
            rank = int(v.shape[0])
            break
    alpha = C.LORA_ALPHA if rank == C.LORA_R else 2 * rank
    if rank != C.LORA_R:
        print(f"  checkpoint LoRA rank {rank} (config default {C.LORA_R}); "
              f"building adapters at the checkpoint's rank, alpha {alpha}", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=C.LORA_DROPOUT,
        target_modules=list(C.LORA_TARGET_MODULES),
    )
    lora_model = get_peft_model(base_model, lora_config)

    # Fail loudly if the checkpoint's hidden dim does not match the configured
    # architecture, instead of silently loading nothing and using random weights.
    for cls_key in ("brainlm.base_model.model.vit.embeddings.cls_token",
                    "base_model.model.vit.embeddings.cls_token"):
        if cls_key in state_dict:
            ckpt_hidden = state_dict[cls_key].shape[-1]
            if ckpt_hidden != C.HIDDEN_SIZE:
                raise ValueError(
                    f"Checkpoint hidden_size={ckpt_hidden} does not match configured "
                    f"HIDDEN_SIZE={C.HIDDEN_SIZE} (src/config.py). "
                    f"'{pretrained_path.name}' is an incompatible model."
                )
            break

    # A stage-1 checkpoint nests the PEFT model under `brainlm.`; strip that one level.
    converted = {k.replace("brainlm.base_model.model.", "base_model.model."): v
                 for k, v in state_dict.items()}
    # `classifier.*` belongs to the stage-1 head, not the backbone.
    n_head = sum(1 for k in converted if k.startswith("classifier."))
    converted = {k: v for k, v in converted.items() if not k.startswith("classifier.")}
    state_dict = converted
    if n_head:
        print(f"  ignoring {n_head} stage-1 classifier tensors (backbone only)", flush=True)

    _report(*lora_model.load_state_dict(converted, strict=False), "LoRA")

    for p in lora_model.parameters():
        p.requires_grad = False
    lora_model.eval()
    return lora_model
