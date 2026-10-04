"""
Token-eviction policies: keep at most `budget` tokens per layer and head.

Memory becomes O(budget) instead of O(T). The price is quality: an evicted
token can never be attended to again. The policies differ only in HOW they
choose which tokens to keep.

| policy          | keeps                                         | needs attention scores |
|-----------------|-----------------------------------------------|------------------------|
| SlidingWindow   | the most recent `budget` tokens               | no                     |
| StreamingLLM    | first `sink` tokens + most recent             | no                     |
| KeyNorm         | tokens whose KEY has the smallest L2 norm     | no                     |
| H2O             | recent + highest ACCUMULATED attention        | yes                    |
| TOVA            | highest attention from the CURRENT query      | yes                    |
| SnapKV          | prompt tokens most attended by the last w     | yes                    |
|                 | prompt queries (pooled), + those w tokens      |                        |

Common options (all eviction policies):
    budget        average tokens per layer (per head)
    sink          first tokens that are never evicted ("attention sinks")
    recent        most recent tokens never evicted; int = count, float < 1 = fraction of budget
    schedule      "uniform" | "pyramid" | "adaptive" (see budgets.py)
    skip_layers   layers that are never compressed (e.g. [0, 1])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import budgets as B
from .base import KVPolicy, gather_tokens, topk_keep

INF = float("inf")


class EvictionPolicy(KVPolicy):
    uses_budget = True

    def __init__(self, budget: int, sink: int = 0, recent: int | float = 0, schedule: str = "uniform",
                 pyramid_ratio: float = 0.5, adaptive_floor: float = 0.25, skip_layers=()):
        if budget is None or budget <= sink:
            raise ValueError(f"{type(self).__name__}: budget ({budget}) must be > sink ({sink})")
        if schedule not in ("uniform", "pyramid", "adaptive"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.avg_budget = int(budget)
        self.sink = int(sink)
        self.recent = recent
        self.schedule = schedule
        self.pyramid_ratio = pyramid_ratio
        self.adaptive_floor = adaptive_floor
        self.skip_layers = set(skip_layers)
        if schedule == "adaptive":
            self.needs_scores = True  # instance attribute overrides the class default
        self._budgets: list[int] | None = None
        self._obs = 1

    @property
    def uniform_lengths(self) -> bool:
        return self.schedule == "uniform" and not self.skip_layers

    def reset(self) -> None:
        self._budgets = None

    def begin_step(self, new_tokens, obs_queries, scores) -> None:
        self._obs = obs_queries
        if self._budgets is not None:
            return
        if self.schedule == "uniform":
            self._budgets = B.uniform(self.avg_budget, self.num_layers)
        elif self.schedule == "pyramid":
            self._budgets = B.pyramid(self.avg_budget, self.num_layers, self.pyramid_ratio)
        elif len(scores) == self.num_layers and max(s.shape[-1] for s in scores.values()) > self.avg_budget:
            # adaptive: decide once, at the first moment eviction becomes necessary
            self._budgets = B.adaptive(self.avg_budget, scores, self.num_layers, self.adaptive_floor)

    def budget_of(self, layer: int) -> int:
        b = self._budgets[layer] if self._budgets else self.avg_budget
        return max(b, self.sink + 1)

    def _recent(self, budget: int) -> int:
        r = int(self.recent * budget) if isinstance(self.recent, float) and self.recent < 1 else int(self.recent)
        return max(0, min(r, budget - self.sink))

    def _protect(self, priority: torch.Tensor, budget: int) -> torch.Tensor:
        """Give sink and recent tokens infinite priority so top-k always keeps them."""
        T = priority.shape[-1]
        r = self._recent(budget)
        if self.sink:
            priority[..., : self.sink] = INF
        if r:
            priority[..., T - r:] = INF
        return priority

    def compress(self, layer, key, value, scores, new_tokens):
        T = key.shape[2]
        if layer in self.skip_layers or T <= self.budget_of(layer):
            return key, value
        return self._evict(layer, key, value, scores, new_tokens, self.budget_of(layer))

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        raise NotImplementedError

    def config(self) -> dict:
        return {"name": self.name, "budget": self.avg_budget, "sink": self.sink, "recent": self.recent,
                "schedule": self.schedule, "skip_layers": sorted(self.skip_layers)}


class SlidingWindow(EvictionPolicy):
    """Keep only the last `budget` tokens. Simplest baseline; loses the beginning
    of the text, which is known to hurt badly (see StreamingLLM)."""

    name = "sliding_window"

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        return key[:, :, -budget:], value[:, :, -budget:]


class StreamingLLM(EvictionPolicy):
    """Xiao et al., ICLR 2024. The first few tokens receive huge attention no matter
    their content ("attention sinks": softmax must put its mass somewhere). Keeping
    them + a recent window fixes most of the sliding-window collapse."""

    name = "streaming_llm"

    def __init__(self, budget: int, sink: int = 4, **kw):
        super().__init__(budget, sink=sink, **kw)

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        s, w = self.sink, budget - self.sink
        return (torch.cat([key[:, :, :s], key[:, :, -w:]], dim=2),
                torch.cat([value[:, :, :s], value[:, :, -w:]], dim=2))


class KeyNorm(EvictionPolicy):
    """Devoto et al., EMNLP 2024: keys with a LOW L2 norm tend to receive HIGH attention.
    Score-free: works with any attention kernel (SDPA, FlashAttention) at zero extra cost."""

    name = "keynorm"

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        priority = self._protect(-key.float().norm(dim=-1), budget)
        idx = topk_keep(priority, budget)
        return gather_tokens(key, idx), gather_tokens(value, idx)


class ScorePolicy(EvictionPolicy):
    """Eviction by attention statistics collected by the kvlab_sdpa backend.

    aggregation:
      "sum"  total attention received so far (H2O). Bias: old tokens had more
             chances to collect attention, so they are favoured.
      "mean" (this lab) sum / number of queries that could see the token.
             Removes the age bias; new tokens compete fairly with old ones.
      "ema"  exponentially decayed sum (decay per forward pass): recent attention counts more.
      "last" only the current forward pass (TOVA).
    """

    needs_scores = True
    name = "score"

    def __init__(self, budget: int, aggregation: str = "sum", decay: float = 0.9, **kw):
        super().__init__(budget, **kw)
        if aggregation not in ("sum", "mean", "ema", "last"):
            raise ValueError(f"unknown aggregation {aggregation!r}")
        self.aggregation = aggregation
        self.decay = decay

    def reset(self) -> None:
        super().reset()
        self.acc: dict[int, torch.Tensor] = {}
        self.seen: dict[int, torch.Tensor] = {}

    @staticmethod
    def _pad(x: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
        if x is None:
            return torch.zeros_like(like)
        extra = like.shape[-1] - x.shape[-1]
        return torch.cat([x, x.new_zeros(*x.shape[:-1], extra)], dim=-1) if extra > 0 else x

    def _update_stats(self, layer: int, scores: torch.Tensor | None) -> None:
        if scores is None:
            raise RuntimeError(f"{self.name} needs attention scores: load the model with "
                               "attn_implementation='kvlab_sdpa' and use KVRunner.")
        if self.aggregation == "last":
            self.acc[layer] = scores
            return
        acc = self._pad(self.acc.get(layer), scores)
        self.acc[layer] = acc * self.decay + scores if self.aggregation == "ema" else acc + scores
        if self.aggregation == "mean":
            T, w = scores.shape[-1], self._obs
            # query r of the observation window sees keys j <= T - w + r -> key j is seen by this many queries
            pos = torch.arange(T, device=scores.device)
            visible = (w - (pos - (T - w)).clamp(min=0)).clamp(min=0).to(scores.dtype)
            self.seen[layer] = self._pad(self.seen.get(layer), scores) + visible

    def _priority(self, layer: int) -> torch.Tensor:
        if self.aggregation == "mean":
            return self.acc[layer] / self.seen[layer].clamp_min(1)
        return self.acc[layer].clone()

    def _keep_stats(self, layer: int, idx: torch.Tensor) -> None:
        self.acc[layer] = self.acc[layer].gather(-1, idx)
        if layer in self.seen:
            self.seen[layer] = self.seen[layer].gather(-1, idx)

    def compress(self, layer, key, value, scores, new_tokens):
        if layer not in self.skip_layers:
            self._update_stats(layer, scores)
            if scores.shape[-1] != key.shape[2]:
                raise RuntimeError(f"layer {layer}: scores cover {scores.shape[-1]} tokens, cache has {key.shape[2]}")
        return super().compress(layer, key, value, scores, new_tokens)

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        idx = topk_keep(self._protect(self._priority(layer), budget), budget)
        self._keep_stats(layer, idx)
        return gather_tokens(key, idx), gather_tokens(value, idx)

    def config(self) -> dict:
        return {**super().config(), "aggregation": self.aggregation}


class H2O(ScorePolicy):
    """Zhang et al., NeurIPS 2023 (Heavy-Hitter Oracle). Half the budget for recent
    tokens, half for the tokens with the largest accumulated attention.

    Note: on a prompt processed in one pass, scores come from the last
    `score_window` queries only (cheap). For EXACT H2O accounting use chunked
    prefill with score_window >= prefill_chunk."""

    name = "h2o"

    def __init__(self, budget: int, recent: int | float = 0.5, **kw):
        super().__init__(budget, recent=recent, **kw)


class TOVA(ScorePolicy):
    """Oren et al., 2024 ("Transformers are Multi-State RNNs"): evict the token with the
    lowest attention from the CURRENT query; no history, no recent window."""

    name = "tova"

    def __init__(self, budget: int, **kw):
        kw.setdefault("aggregation", "last")
        super().__init__(budget, **kw)


class SnapKV(EvictionPolicy):
    """Li et al., 2024. Compress the PROMPT once: the last `w` prompt tokens (the
    observation window, usually containing the question) vote for which earlier
    tokens matter. Votes are smoothed with 1-D pooling so neighbouring tokens are
    kept together (keeps phrases intact instead of isolated tokens).
    Generated tokens are not evicted (paper setting), so memory grows slowly while decoding.
    The observation window size = the runner's score_window."""

    name = "snapkv"
    needs_scores = True

    def __init__(self, budget: int, kernel: int = 7, pooling: str = "max", **kw):
        super().__init__(budget, **kw)
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd so pooling keeps the sequence length")
        self.kernel, self.pooling = kernel, pooling

    def compress(self, layer, key, value, scores, new_tokens):
        if new_tokens <= 1:
            return key, value
        return super().compress(layer, key, value, scores, new_tokens)

    def _evict(self, layer, key, value, scores, new_tokens, budget):
        T, w = key.shape[2], self._obs
        n_prefix = budget - w
        if n_prefix <= 0:  # budget smaller than the window: keep the newest tokens only
            return key[:, :, -budget:], value[:, :, -budget:]
        votes = scores[..., : T - w].float()
        b, h, n = votes.shape
        flat = votes.reshape(b * h, 1, n)
        pad = self.kernel // 2
        if self.pooling == "max":
            flat = F.max_pool1d(flat, self.kernel, stride=1, padding=pad)
        else:
            flat = F.avg_pool1d(flat, self.kernel, stride=1, padding=pad, count_include_pad=False)
        votes = flat.reshape(b, h, n)
        if self.sink:
            votes[..., : self.sink] = INF
        idx_prefix = topk_keep(votes, n_prefix)
        idx_window = torch.arange(T - w, T, device=key.device).expand(b, h, w)
        idx = torch.cat([idx_prefix, idx_window], dim=-1)
        return gather_tokens(key, idx), gather_tokens(value, idx)

    def config(self) -> dict:
        return {**super().config(), "kernel": self.kernel, "pooling": self.pooling}
