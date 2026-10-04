"""
Base class and tensor helpers for KV-cache policies.

What is a policy?
-----------------
After EVERY forward pass (a prefill chunk or one decode step) the runner calls

    key, value = policy.compress(layer, key, value, scores, new_tokens)

for every layer. `key`/`value` are that layer's full cache, shape
(B, Hkv, T, Dh), already including the tokens just processed. A policy may
return them unchanged (full cache), return fewer tokens (eviction), or
overwrite some values with lower-precision versions (quantization).

Invariants every policy must keep
---------------------------------
1. Temporal order: kept tokens stay in their original relative order and the
   newest tokens stay at the END. The causal mask during chunked prefill
   relies on "the last Tq entries are the current chunk".
2. No re-rotation: keys were rotated with RoPE at their ORIGINAL absolute
   position. We never change that; the runner keeps feeding absolute
   positions to new tokens. (StreamingLLM re-assigns positions inside the
   cache to extrapolate beyond the trained length. Llama-3 supports 128k, so we
   stay within trained positions and avoid that complication.)
3. Per head, every layer keeps the same number of tokens for all heads
   (different WHICH tokens is fine). The tensor must stay rectangular.
"""

from __future__ import annotations

import torch


class KVPolicy:
    """Full cache: keeps everything. Also the base class of all policies."""

    name = "full"
    needs_scores = False  # True -> requires the kvlab_sdpa attention backend
    uses_budget = False

    @property
    def uniform_lengths(self) -> bool:
        """False if layers may hold different numbers of tokens (needs kvlab_sdpa)."""
        return True

    def setup(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self.reset()

    def reset(self) -> None:
        """Clear per-sequence state. Called before every new prompt."""

    def begin_step(self, new_tokens: int, obs_queries: int, scores: dict[int, torch.Tensor]) -> None:
        """Called once per forward pass, before the per-layer compress() calls.

        new_tokens  : tokens processed in this forward (chunk size, or 1 when decoding)
        obs_queries : how many of the newest queries contributed to `scores`
        scores      : {layer: (B, Hkv, T)} attention mass per cached token (may be empty)
        """

    def compress(self, layer: int, key: torch.Tensor, value: torch.Tensor,
                 scores: torch.Tensor | None, new_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        return key, value

    def layer_logical_bytes(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> float:
        """Bytes a real implementation would need. Equals the tensor size unless
        the policy SIMULATES a smaller format (quantization)."""
        return key.numel() * key.element_size() + value.numel() * value.element_size()

    def logical_bytes(self, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        return sum(self.layer_logical_bytes(i, k, v) for i, (k, v) in enumerate(layers))

    def config(self) -> dict:
        return {"name": self.name}


FullCache = KVPolicy


def gather_tokens(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Select tokens per head. x: (B, H, T, D), idx: (B, H, n) -> (B, H, n, D)."""
    return x.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))


def topk_keep(priority: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Indices of the n_keep highest-priority tokens per head, sorted by position."""
    return priority.topk(n_keep, dim=-1).indices.sort(dim=-1).values
