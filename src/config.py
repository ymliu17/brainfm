"""Central experiment configuration (single source of truth).

The backbone architecture is the released BrainLM 13M: hidden 512, 4 layers, 4 heads,
intermediate 1024. Verified against `vandijklab/brainlm` `old_13M/config.json` and its
weights. `model_loader` refuses a checkpoint whose shapes disagree.

There is deliberately no profile switch any more. Earlier versions of this file offered
several architectures, and a 256-dim setting that turned out not to be BrainLM at all was
selected by default -- which silently invalidated every stage-1 result produced under it.
One architecture, checked against the released weights, removes that failure mode.
"""
from __future__ import annotations

import os

# --- Foundation-model architecture: the released BrainLM 13M ---
HIDDEN_SIZE = 512
DECODER_HIDDEN_SIZE = 512
BATCH_SIZE = 20

# Random patch masking is a pretraining/fine-tuning augmentation only. The encoder
# applies it in training mode and never in eval mode, so downstream CLS extraction
# (which always runs the frozen backbone in eval) is deterministic regardless of
# this value. See BrainLMEmbeddings.forward.
MASK_RATIO = 0.2

# --- Remaining architecture constants ---
NUM_HIDDEN_LAYERS = 4
NUM_ATTENTION_HEADS = 4
INTERMEDIATE_SIZE = 1024
DECODER_NUM_HIDDEN_LAYERS = 2
DECODER_NUM_ATTENTION_HEADS = 4
DECODER_INTERMEDIATE_SIZE = 1024
NUM_BRAIN_VOXELS = 424

# BrainLM's pretraining window: 200 frames at TR 0.735 s, i.e. 200/20 = 10 time patches
# per ROI and a sequence of 1 + 424*10 = 4241 tokens. This is NOT an inert field -- an
# earlier comment here called it that. The official inference pipeline cuts every
# recording into non-overlapping 200-frame windows and averages the CLS token across
# them (see src/brainlm_input.py).
NUM_TIMEPOINTS_PER_VOXEL = 200
TIMEPOINT_PATCHING_SIZE = 20

# --- LoRA (q/k/v + output projection of self-attention) ---
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "key", "value", "dense"]

# --- Downstream linear-probing training ---
EPOCHS = 20
LEARNING_RATE = 1e-3
SEED = 42

# Reported results average over these seeds. Single-seed numbers are not stable at
# this sample size: on act/concat, seed 42 and seed 43 differ by 0.05 AUC, which is
# larger than the across-fold standard deviation.
SEEDS = (42, 43, 44, 45, 46)

# Fraction of each outer training fold held out to choose the training epoch. The
# outer fold is never used for model selection -- it is scored exactly once.
INNER_VAL_FRAC = 0.2

# Penultimate-layer width of the downstream classifier head = the d=128 embedding
# matrix X used in the PLS analysis (Methods, Q4).
CLASSIFIER_EMBED_DIM = 128

# --- Numerics ---
# TF32 makes the frozen-backbone forward ~1.75x faster on A100 (measured, batch 20,
# ACT) at ~1e-3 absolute error on the CLS features. It is off by default so the
# numbers in the original cohort audit reproduce exactly; turn it on with BRAINLM_TF32=1
# for exploratory runs. The setting is part of the feature-cache key, so fp32 and
# tf32 caches can never be mixed.
ALLOW_TF32 = os.environ.get("BRAINLM_TF32", "0") not in ("0", "", "false", "False")

# Fused scaled-dot-product attention in the encoder. Same function as the eager
# path (see brainlm_mae/fast_attention.py for the equivalence argument), but it never
# materialises the n x n score matrix, which for BrainLM's 2969-6361 token sequences
# is what actually caps the batch size. Measured on A100 / ACT:
#   inference  batch 20: 7.41 s -> 5.01 s, 28.9 GB -> 3.0 GB; batch 95 now fits
#   training   batch  8: 0.178 -> 0.103 s/step, 9.8 GB -> 2.5 GB
#   agreement  2.0e-5 max abs on CLS features (fp32 reassociation)
# On by default; BRAINLM_FAST_ATTN=0 restores the eager path bit-for-bit.
FAST_ATTENTION = os.environ.get("BRAINLM_FAST_ATTN", "1") not in ("0", "", "false", "False")


def configure_backends() -> None:
    """Apply the numeric and kernel settings. Call once at the top of an entrypoint."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = ALLOW_TF32

    from brainlm_mae.fast_attention import disable_sdpa_attention, enable_sdpa_attention

    if FAST_ATTENTION:
        enable_sdpa_attention(verbose=False)
    else:
        disable_sdpa_attention()


def summary() -> str:
    return (
        f"brainlm13m | hidden={HIDDEN_SIZE} decoder={DECODER_HIDDEN_SIZE} "
        f"train_mask_ratio={MASK_RATIO} batch={BATCH_SIZE} tf32={ALLOW_TF32} "
        f"fast_attn={FAST_ATTENTION}"
    )
