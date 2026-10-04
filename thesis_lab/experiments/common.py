"""
Helpers shared by all experiments.

Run every experiment from the thesis_lab/ folder as a module, e.g.
    cd thesis_lab
    python -m experiments.e01_analytic_memory
so that `import kvlab` works without installing anything.
"""

from __future__ import annotations

import argparse
import random

import torch

from kvlab import data, io, measure, models


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="path to a YAML config (default configs/lab.yaml)")
    p.add_argument("--model", default=None, help="override the model id from the config")
    p.add_argument("--quick", action="store_true", help="tiny sweep: checks the pipeline end-to-end in minutes")
    return p.parse_args()


def setup(args: argparse.Namespace) -> dict:
    cfg = io.load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    return cfg


def load_model(cfg: dict, attn: str):
    return models.load(cfg["model"], cfg["dtype"], cfg["device"], attn=attn)


def text_ids(cfg: dict, tokenizer, lang: str, n_tokens: int, device) -> torch.Tensor:
    # ~6 characters per token is a safe upper bound for English; other scripts need fewer.
    text = data.load_text(lang, min_chars=max(400_000, 6 * n_tokens),
                          source=cfg["data"]["source"], file=cfg["data"]["file"])
    return data.prompt_ids(tokenizer, text, n_tokens, device)


def oom_safe(fn, *args, **kwargs):
    """Run fn; on CUDA out-of-memory return the string "OOM" instead of crashing the sweep.
    Long-context sweeps are EXPECTED to hit OOM at some length - that point is a result."""
    try:
        return fn(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        measure.free_memory()
        return "OOM"
