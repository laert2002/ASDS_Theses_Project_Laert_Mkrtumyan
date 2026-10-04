"""
Timing and memory measurement helpers.

Two rules for correct GPU measurements:

1. CUDA is asynchronous: a Python line like `model(x)` returns BEFORE the GPU
   has finished. Without torch.cuda.synchronize() you would time only the
   kernel *launch*. `Timer` synchronises on enter and exit.

2. "Peak memory" = torch.cuda.max_memory_allocated(): the maximum bytes held
   by live tensors since the last reset. It includes weights + KV cache +
   temporary activations (e.g. attention buffers, logits). It excludes memory
   the caching allocator reserved but did not hand out (fragmentation), which
   nvidia-smi would show. For thesis tables, "allocated" is the standard and
   reproducible choice; mention this in the methodology section.
"""

from __future__ import annotations

import gc
import time

import torch

MiB = 1024**2
GiB = 1024**3


def _is_cuda(device) -> bool:
    return torch.device(device).type == "cuda"


def sync(device) -> None:
    if _is_cuda(device):
        torch.cuda.synchronize(device)


def free_memory() -> None:
    """Release Python garbage and cached CUDA blocks (call between experiments)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_peak(device) -> None:
    if _is_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def peak_bytes(device) -> int | None:
    return torch.cuda.max_memory_allocated(device) if _is_cuda(device) else None


def allocated_bytes(device) -> int | None:
    return torch.cuda.memory_allocated(device) if _is_cuda(device) else None


class Timer:
    """Wall-clock timer that is correct for asynchronous CUDA work.

    with Timer(device) as t:
        model(x)
    print(t.seconds)
    """

    def __init__(self, device):
        self.device = device
        self.seconds = float("nan")

    def __enter__(self):
        sync(self.device)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.seconds = time.perf_counter() - self._t0
        return False


def to_gib(x: float | None) -> float | None:
    return None if x is None else x / GiB


def to_mib(x: float | None) -> float | None:
    return None if x is None else x / MiB
