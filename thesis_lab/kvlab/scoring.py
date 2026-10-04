"""
Custom attention backend "kvlab_sdpa".

Problem it solves
-----------------
Score-based KV-cache policies (H2O, SnapKV, mass allocation, tiered precision)
need to know HOW MUCH ATTENTION each cached token receives. The fast kernels
(SDPA / FlashAttention) never expose the attention matrix, and
output_attentions=True forces "eager" attention which materialises the full
(B, H, T, T) matrix -> impossible at long context (see docs/shapes.md).

Solution
--------
transformers (>= 4.48) lets you register your own attention function through
`AttentionInterface`. Ours:
  1. computes the attention OUTPUT with torch's fused SDPA (fast, O(T) memory);
  2. if recording is enabled, recomputes softmax(q K^T) ONLY for the last `w`
     queries ("observation window", SnapKV idea). Cost: O(w * T) instead of
     O(T^2). During decoding w = 1, so it is almost free.
The per-token attention mass is stored in the global `STORE`, where the
runner hands it to the active policy after every forward pass.

A second benefit: we build the causal mask ourselves ("bottom-right aligned")
instead of trusting the mask HF builds from layer 0. That makes it correct even
when layers keep DIFFERENT numbers of tokens (per-layer budgets), which the
stock implementation does not support during chunked prefill.

Assumption: batch size 1 without padding (true for all experiments here).
Padding support would require honouring `attention_mask`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

ATTN_NAME = "kvlab_sdpa"

_TORCH_VERSION = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
_HAS_SDPA_GQA = _TORCH_VERSION >= (2, 5)  # `enable_gqa` argument of SDPA


class ScoreStore:
    """Holds attention statistics of the most recent forward pass, per layer.

    scores[layer] : (B, Hkv, T_cache) float32
        attention probability mass each cached token received, summed over the
        observation-window queries and over the query heads that share the
        KV head (GQA group). Each head's row therefore sums to group * w.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.window = 32
        self.scores: dict[int, torch.Tensor] = {}

    def configure(self, enabled: bool, window: int) -> None:
        self.enabled, self.window = enabled, window

    def clear(self) -> None:
        self.scores = {}

    @torch.no_grad()
    def record(self, layer: int, query: torch.Tensor, key: torch.Tensor, scale: float) -> None:
        B, Hq, Tq, D = query.shape
        Hkv, Tk = key.shape[1], key.shape[2]
        g = Hq // Hkv
        w = min(self.window, Tq)

        # Query head h uses KV head h // g, so (Hq) can be viewed as (Hkv, g).
        # Matmul per KV head avoids materialising g copies of the key cache.
        q = query[:, :, -w:, :].reshape(B, Hkv, g * w, D)
        logits = (q @ key.transpose(-1, -2)).float().mul_(scale).view(B, Hkv, g, w, Tk)

        # Observation row r is chunk position Tq - w + r -> it may see keys up to Tk - w + r.
        limit = Tk - w + torch.arange(w, device=key.device)
        future = torch.arange(Tk, device=key.device)[None, :] > limit[:, None]  # (w, Tk)
        logits.masked_fill_(future, float("-inf"))

        self.scores[layer] = logits.softmax(dim=-1).sum(dim=(2, 3))


STORE = ScoreStore()


def _repeat_kv(x: torch.Tensor, g: int) -> torch.Tensor:
    return x if g == 1 else x.repeat_interleave(g, dim=1)


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Causal SDPA where the Tq queries are the LAST Tq positions of the Tk keys."""
    Tq, Tk = q.shape[2], k.shape[2]
    g = q.shape[1] // k.shape[1]

    if Tq == 1 or Tq == Tk:
        # decode step (sees everything) or first prefill (square causal): no explicit mask needed
        causal = Tq > 1
        if _HAS_SDPA_GQA:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale, enable_gqa=g > 1)
        return F.scaled_dot_product_attention(q, _repeat_kv(k, g), _repeat_kv(v, g), is_causal=causal, scale=scale)

    # Chunk attending to a longer cache (chunked prefill / multi-token scoring).
    # causal_lower_right lets SDPA pick a memory-efficient kernel instead of a (Tq, Tk) bool mask.
    # Repeating K/V costs one temporary copy of this layer's cache (x group size).
    from torch.nn.attention.bias import causal_lower_right

    return F.scaled_dot_product_attention(
        q, _repeat_kv(k, g), _repeat_kv(v, g), attn_mask=causal_lower_right(Tq, Tk), scale=scale
    )


def kvlab_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """Signature required by transformers' AttentionInterface.

    query : (B, Hq,  Tq, D)  already rotated with RoPE
    key   : (B, Hkv, Tk, D)  full cache for this layer INCLUDING the new tokens
    value : (B, Hkv, Tk, D)
    returns (attn_output (B, Tq, Hq, D), attn_weights=None)
    """
    scale = scaling if scaling is not None else 1.0 / math.sqrt(query.shape[-1])
    out = _attend(query, key, value, scale)
    if STORE.enabled:
        STORE.record(module.layer_idx, query, key, scale)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    # kvlab_attention builds its own causal mask; skip HF's (Tq x Tk) mask allocation.
    return None


_REGISTERED = False


def register() -> str:
    """Register "kvlab_sdpa" with transformers. Call BEFORE loading the model."""
    global _REGISTERED
    if not _REGISTERED:
        from transformers import AttentionInterface

        AttentionInterface.register(ATTN_NAME, kvlab_attention)
        try:
            from transformers.masking_utils import AttentionMaskInterface
        except ImportError:  # transformers < 4.53 builds a generic mask; still correct, uses more memory
            AttentionMaskInterface = None
        if AttentionMaskInterface is not None:
            AttentionMaskInterface.register(ATTN_NAME, _no_mask)
        _REGISTERED = True
    return ATTN_NAME
