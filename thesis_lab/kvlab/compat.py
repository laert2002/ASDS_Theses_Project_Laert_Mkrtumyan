"""
Version-independent access to the Hugging Face KV cache.

The cache object returned in `out.past_key_values` changed its internals
several times. Policies in this lab only need three operations:
    * read  (key, value) of every layer,
    * replace (key, value) of a layer after compression,
    * measure its size.
Isolating them here means a future transformers upgrade touches ONE file.

Tensor layout (all versions): key/value = (batch, num_kv_heads, seq_len, head_dim).
"""

from __future__ import annotations

import torch

KV = tuple[torch.Tensor, torch.Tensor]


def new_cache():
    """Empty DynamicCache (grows by concatenation along the sequence axis).

    Note: DynamicCache appends with torch.cat, i.e. it copies the whole layer
    cache on every step (O(T) memory traffic per token). A StaticCache avoids
    this by pre-allocating; see experiments/e07 for a comparison.
    """
    from transformers import DynamicCache

    return DynamicCache()


def cache_layers(cache) -> list[KV]:
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):  # 4.36 - 4.55
        return list(zip(cache.key_cache, cache.value_cache))
    return [(k, v) for k, v in cache]  # legacy tuple-of-tuples


def set_layer(cache, idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[idx].keys = key
        cache.layers[idx].values = value
    elif hasattr(cache, "key_cache"):
        cache.key_cache[idx] = key
        cache.value_cache[idx] = value
    else:
        raise TypeError("Legacy tuple caches are immutable; use transformers >= 4.56.")


def layer_lengths(cache) -> list[int]:
    return [k.shape[2] for k, _ in cache_layers(cache)]


def cache_bytes(cache) -> int:
    """Bytes actually held on the device, counted per unique storage.

    Slicing (e.g. a sliding window) returns a VIEW that keeps the whole old
    storage alive until the next update replaces it, so numel() would under-report.
    """
    seen, total = set(), 0
    for k, v in cache_layers(cache):
        for t in (k, v):
            storage = t.untyped_storage()
            if storage.data_ptr() not in seen:
                seen.add(storage.data_ptr())
                total += storage.nbytes()
    return total
