"""
Per-layer budget schedules: how many tokens each layer may keep.

A single number "budget = 1024" can be spread over layers in different ways
while the TOTAL memory stays the same (sum of budgets = L * average):

uniform   every layer keeps the same number of tokens (most papers).

pyramid   lower layers keep more, upper layers keep fewer (PyramidKV, Cai et
          al. 2024). Motivation: lower layers attend broadly, upper layers
          concentrate on few tokens ("information funnelling").

adaptive  (this lab) budget proportional to the EFFECTIVE NUMBER OF ATTENDED
          TOKENS of each layer, exp(entropy of its attention distribution),
          measured on the prompt. A layer that spreads attention over 3,000
          tokens gets more room than one that looks at 50. Related to CAKE
          (Qin et al., ICLR 2025) which uses entropy + variance. This is a
          simple, explainable variant that works without any tuning.
"""

from __future__ import annotations

import torch


def uniform(avg: int, num_layers: int) -> list[int]:
    return [avg] * num_layers


def pyramid(avg: int, num_layers: int, ratio: float = 0.5) -> list[int]:
    """Linear from avg*(1+ratio) at layer 0 to avg*(1-ratio) at the last layer; mean = avg."""
    if num_layers == 1:
        return [avg]
    return [max(1, round(avg * (1 + ratio - 2 * ratio * i / (num_layers - 1)))) for i in range(num_layers)]


def effective_tokens(scores: torch.Tensor) -> float:
    """exp(entropy) of the attention distribution, averaged over heads. scores: (B, H, T)."""
    p = scores.float() / scores.float().sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1)
    return entropy.exp().mean().item()


def adaptive(avg: int, scores_by_layer: dict[int, torch.Tensor], num_layers: int,
             floor: float = 0.25, cap: float = 4.0) -> list[int]:
    """Budgets proportional to effective_tokens, clamped to [floor*avg, cap*avg], mean ~ avg."""
    eff = torch.tensor([effective_tokens(scores_by_layer[i]) for i in range(num_layers)])
    raw = (eff / eff.mean() * avg).clamp(floor * avg, cap * avg)
    raw = raw * (avg * num_layers / raw.sum())  # clamping shifts the total; restore it
    return [max(1, round(x)) for x in raw.tolist()]
