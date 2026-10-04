"""
Precision-reduction policies: keep ALL tokens, but store some with fewer bits.

SIMULATED quantization (important for your methodology section)
---------------------------------------------------------------
These policies quantize and immediately de-quantize values in place
("fake quantization"). The model then sees exactly the rounding error a real
int4/int2 cache would produce, so QUALITY results are exact. The tensors still
live in fp16, so MEASURED GPU memory does not drop. The memory the method would
need is reported analytically via `layer_logical_bytes`.

For real memory savings, use the HF quantized cache (experiments/e07), or
implement packed storage + a dequantizing attention kernel (future work).

Uniform asymmetric quantization of a group x_1..x_g to b bits:
    scale = (max - min) / (2^b - 1)
    q_i   = round((x_i - min) / scale)      integer in [0, 2^b - 1]
    x^_i  = q_i * scale + min               what the model sees
Each group also stores min and scale in fp16 (the overhead in memory_model).

Which axis to group along (KIVI, Liu et al., ICML 2024)?
    KEYS have a few outlier CHANNELS with huge magnitude -> group per channel
         (across tokens), so one outlier channel does not destroy the others.
    VALUES have no channel structure -> group per token (across channels).
"""

from __future__ import annotations

import torch

from ..memory_model import kv_bits_per_element
from .base import KVPolicy
from .eviction import ScorePolicy

INF = float("inf")


def fake_quant(x: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Quantize-dequantize with min/max taken along `dim` (the group axis)."""
    xf = x.float()
    lo = xf.amin(dim=dim, keepdim=True)
    hi = xf.amax(dim=dim, keepdim=True)
    levels = 2**bits - 1
    scale = (hi - lo).clamp_min(1e-8) / levels
    q = ((xf - lo) / scale).round_().clamp_(0, levels)
    return (q * scale + lo).to(x.dtype)


def quant_per_channel(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """x: (B, H, n*group, D). Groups of `group` consecutive tokens, per channel."""
    b, h, t, d = x.shape
    return fake_quant(x.reshape(b, h, t // group, group, d), bits, dim=3).reshape(b, h, t, d)


def quant_per_token(x: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """x: (B, H, T, D) with D % group == 0. Groups of `group` channels, per token."""
    b, h, t, d = x.shape
    return fake_quant(x.reshape(b, h, t, d // group, group), bits, dim=4).reshape(b, h, t, d)


def _elements(key: torch.Tensor) -> int:
    """Numbers per token per tensor (K or V) in one layer: B * Hkv * Dh."""
    return key.shape[0] * key.shape[1] * key.shape[3]


class KIVI(KVPolicy):
    """Keys per-channel, values per-token, the newest `residual` tokens stay in full
    precision (they are attended most and per-channel groups need `group` tokens to form).
    Tokens are quantized in blocks of `group_size` once they leave the residual window."""

    name = "kivi"

    def __init__(self, bits: int = 4, group_size: int = 32, residual: int = 128):
        self.bits, self.group, self.residual = bits, group_size, residual

    def reset(self) -> None:
        self.n_quant: dict[int, int] = {}

    def compress(self, layer, key, value, scores, new_tokens):
        T, D = key.shape[2], key.shape[3]
        if D % self.group:
            raise ValueError(f"head_dim {D} not divisible by group_size {self.group}")
        start = self.n_quant.get(layer, 0)
        ready = (T - self.residual - start) // self.group * self.group
        if ready > 0:
            end = start + ready
            key[:, :, start:end] = quant_per_channel(key[:, :, start:end], self.bits, self.group)
            value[:, :, start:end] = quant_per_token(value[:, :, start:end], self.bits, self.group)
            self.n_quant[layer] = end
        return key, value

    def layer_logical_bytes(self, layer, key, value) -> float:
        T, n_q = key.shape[2], self.n_quant.get(layer, 0)
        bits = n_q * kv_bits_per_element(self.bits, self.group) + (T - n_q) * key.element_size() * 8
        return 2 * _elements(key) * bits / 8

    def config(self) -> dict:
        return {"name": self.name, "bits": self.bits, "group_size": self.group, "residual": self.residual}


class Tiered(ScorePolicy):
    """Mixed precision instead of eviction ("no token left behind", cf. MiKV, Yang et al. 2024).

    `budget` tokens per head stay in full precision: sinks, the recent window and the
    most-attended tokens. Every other token is DEMOTED to `low_bits` instead of being
    deleted. Eviction loses a token forever; demotion keeps a blurred copy, so
    information that becomes relevant later (e.g. a passkey) can still be found.

    Memory = budget * 16 bits + (T - budget) * low_bits(+overhead): still O(T), but
    with a 3-8x smaller slope. Tokens are demoted individually and per head, so
    both K and V use per-token groups. Keys would prefer per-channel groups (see
    KIVI); combining the two is a good follow-up experiment.
    """

    name = "tiered"

    def __init__(self, budget: int, low_bits: int = 4, group_size: int = 32, sink: int = 4,
                 recent: int | float = 0.25, aggregation: str = "mean", **kw):
        super().__init__(budget, sink=sink, recent=recent, aggregation=aggregation, **kw)
        self.low_bits, self.group = low_bits, group_size

    @property
    def uniform_lengths(self) -> bool:
        return True  # nothing is removed, all layers keep every token

    def reset(self) -> None:
        super().reset()
        self.low: dict[int, torch.Tensor] = {}

    def compress(self, layer, key, value, scores, new_tokens):
        if layer in self.skip_layers:
            return key, value
        self._update_stats(layer, scores)
        T, D = key.shape[2], key.shape[3]
        low = self.low.get(layer)
        low = torch.zeros(scores.shape, dtype=torch.bool, device=key.device) if low is None else \
            torch.cat([low, low.new_zeros(*low.shape[:-1], T - low.shape[-1])], dim=-1)
        budget = self.budget_of(layer)
        n_demote = T - int(low[0, 0].sum()) - budget  # same count for every head by construction
        if n_demote > 0:
            priority = self._protect(self._priority(layer), budget).masked_fill(low, INF)
            idx = priority.topk(n_demote, dim=-1, largest=False).indices
            e = idx.unsqueeze(-1).expand(-1, -1, -1, D)
            key.scatter_(2, e, quant_per_token(key.gather(2, e), self.low_bits, self.group))
            value.scatter_(2, e, quant_per_token(value.gather(2, e), self.low_bits, self.group))
            low.scatter_(2, idx, True)
        self.low[layer] = low
        return key, value

    def layer_logical_bytes(self, layer, key, value) -> float:
        T = key.shape[2]
        n_low = int(self.low[layer][0, 0].sum()) if layer in self.low else 0
        bits = n_low * kv_bits_per_element(self.low_bits, self.group) + (T - n_low) * key.element_size() * 8
        return 2 * _elements(key) * bits / 8

    def config(self) -> dict:
        return {**super().config(), "low_bits": self.low_bits, "group_size": self.group}
