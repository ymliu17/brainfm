"""Memory-efficient attention for the BrainLM encoder, without changing the math.

Why this exists
---------------
`BrainLMConfig` sets `num_landmarks == segment_means_seq_len == 64`, and HF's
`NystromformerSelfAttention.forward` branches on exactly that equality, so BrainLM
always takes the **full quadratic attention** path. The Nystrom approximation is not
merely unused, it is unusable here: its reshape divides by `self.seq_len` taken from
the config, which requires the real sequence length to be fixed and divisible by
`num_landmarks`. BrainLM's sequences are 424 ROIs x (T/20) time patches plus a CLS
token -- 6361 for ACT, 2121 for CogTE -- neither fixed nor divisible by 64.

So the cost is not an approximation that was switched off; it is genuine O(n^2)
attention. The score tensor alone is `B x heads x n^2 x 4 B`: 12.9 GB at batch 20 for
ACT, and batch 48 needs 28.9 GB and OOMs a 40 GB A100.

The fix does not touch the model's mathematics. In that branch,
`query`/`key` are each pre-divided by `sqrt(sqrt(head_dim))`, so the branch computes
plain softmax(QK^T / sqrt(d)) V with no attention dropout -- i.e. exactly
`scaled_dot_product_attention(q, k, v, scale=1.0)`. PyTorch's SDPA computes the same
function with fused kernels that never materialise the n x n matrix, so memory drops
from O(B H n^2) to O(B H n d) and existing pretrained weights stay valid.

Usage:
    from brainlm_mae.fast_attention import enable_sdpa_attention
    enable_sdpa_attention()          # process-wide, before building the model

Falls back to the original implementation whenever it cannot be exact:
`output_attentions=True` (SDPA does not return the probability matrix) or a config
where the Nystrom branch would actually be taken.
"""
from __future__ import annotations

import torch.nn as nn
from transformers.models.nystromformer import modeling_nystromformer as _mn

_ORIGINAL_FORWARD = _mn.NystromformerSelfAttention.forward
_PATCHED = False


def _split_heads(self, tensor):
    """[B, T, H*d] -> [B, H, T, d].

    transformers <= 4.46 exposed this as `transpose_for_scores`; 4.57 inlined it as
    `.view(...).transpose(1, 2)`. Support both so the patch survives the upgrade.
    """
    fn = getattr(self, "transpose_for_scores", None)
    if fn is not None:
        return fn(tensor)
    batch_size = tensor.shape[0]
    return tensor.view(
        batch_size, -1, self.num_attention_heads, self.attention_head_size
    ).transpose(1, 2)


def _sdpa_forward(self, hidden_states, attention_mask=None, output_attentions=False):
    # Anything the fused kernel cannot reproduce exactly goes to the original code.
    if output_attentions or self.num_landmarks != self.seq_len:
        return _ORIGINAL_FORWARD(self, hidden_states, attention_mask, output_attentions)

    query_layer = _split_heads(self, self.query(hidden_states))
    key_layer = _split_heads(self, self.key(hidden_states))
    value_layer = _split_heads(self, self.value(hidden_states))

    # The original halves the scaling across q and k; SDPA wants the total factor,
    # so pre-scale identically and pass scale=1.0.
    inv_sqrt_sqrt = 1.0 / (self.attention_head_size ** 0.25)
    query_layer = query_layer * inv_sqrt_sqrt
    key_layer = key_layer * inv_sqrt_sqrt

    context_layer = nn.functional.scaled_dot_product_attention(
        query_layer,
        key_layer,
        value_layer,
        attn_mask=attention_mask,
        dropout_p=0.0,  # the eager branch applies no dropout to attention probs
        scale=1.0,
    )

    if self.conv_kernel_size is not None:
        context_layer = context_layer + self.conv(value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    context_layer = context_layer.view(*context_layer.size()[:-2], self.all_head_size)
    return (context_layer,)


def enable_sdpa_attention(verbose: bool = True) -> None:
    global _PATCHED
    if _PATCHED:
        return
    _mn.NystromformerSelfAttention.forward = _sdpa_forward
    _PATCHED = True
    if verbose:
        print("BrainLM attention: fused SDPA enabled (same math, no n^2 buffer)", flush=True)


def disable_sdpa_attention(verbose: bool = False) -> None:
    global _PATCHED
    if not _PATCHED:
        return
    _mn.NystromformerSelfAttention.forward = _ORIGINAL_FORWARD
    _PATCHED = False
    if verbose:
        print("BrainLM attention: restored the original eager implementation", flush=True)


def is_enabled() -> bool:
    return _PATCHED
