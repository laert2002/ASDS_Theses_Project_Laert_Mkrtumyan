"""
KV-cache policies and a factory that builds them from config dictionaries.

    from kvlab.policies import build
    policy = build({"type": "h2o", "recent": 0.5}, budget=1024)

Adding a new policy:
    1. subclass KVPolicy (or EvictionPolicy / ScorePolicy) and implement compress()/_evict()
    2. add it to REGISTRY below
    3. reference it by name in configs/lab.yaml -> every experiment can use it
"""

from __future__ import annotations

from .base import FullCache, KVPolicy
from .eviction import H2O, TOVA, EvictionPolicy, KeyNorm, ScorePolicy, SlidingWindow, SnapKV, StreamingLLM
from .quant import KIVI, Tiered

REGISTRY: dict[str, type[KVPolicy]] = {
    "full": FullCache,
    "sliding_window": SlidingWindow,
    "streaming_llm": StreamingLLM,
    "keynorm": KeyNorm,
    "h2o": H2O,
    "tova": TOVA,
    "snapkv": SnapKV,
    "kivi": KIVI,
    "tiered": Tiered,
}


def build(spec: dict, budget: int | None = None) -> KVPolicy:
    spec = dict(spec)
    cls = REGISTRY[spec.pop("type")]
    return cls(budget=budget, **spec) if cls.uses_budget else cls(**spec)


__all__ = ["REGISTRY", "build", "KVPolicy", "FullCache", "EvictionPolicy", "ScorePolicy", "SlidingWindow",
           "StreamingLLM", "KeyNorm", "H2O", "TOVA", "SnapKV", "KIVI", "Tiered"]
