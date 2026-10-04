"""
Quality metrics under a KV-cache policy.

Perplexity (PPL) = exp(mean NLL per token). Lower = better. Standard for
measuring how much a cache compression method damages language modelling.

Bits per byte (BPB) = total NLL in bits / UTF-8 bytes of the text.
    PPL is NOT comparable across languages: the same sentence is 1 token in
    one language and 5 in another (see docs/tokenization.md). BPB divides by the
    size of the TEXT, not the number of tokens, so English vs Armenian
    comparisons become fair. Use BPB for every cross-lingual claim.

Passkey accuracy: fraction of prompts where the generated continuation
contains the hidden number.
"""

from __future__ import annotations

import math
import random

from . import data
from .runner import KVRunner


def perplexity(runner: KVRunner, input_ids, chunk: int = 32, tokenizer=None) -> dict:
    nll = runner.nll(input_ids, chunk=chunk)
    n = nll.numel()
    half = n // 2
    res = {
        "eval_tokens": n,
        "eval_chunk": chunk,
        "ppl": math.exp(nll.mean().item()),
        # second half = positions far beyond a small budget: where eviction hurts most
        "ppl_first_half": math.exp(nll[:half].mean().item()),
        "ppl_second_half": math.exp(nll[half:].mean().item()),
    }
    if tokenizer is not None:
        n_bytes = len(tokenizer.decode(input_ids[0, 1:], skip_special_tokens=True).encode("utf-8"))
        res["text_bytes"] = n_bytes
        res["bpb"] = nll.sum().item() / (math.log(2) * n_bytes)
    stats = runner.cache_stats()
    res["cache_logical_bytes"] = stats["logical_bytes"]
    res["cache_held_bytes"] = stats["held_bytes"]
    res["cache_tokens_mean"] = sum(stats["cache_tokens_per_layer"]) / len(stats["cache_tokens_per_layer"])
    return res


def passkey_trial(runner: KVRunner, tokenizer, n_tokens: int, depth: float, key: int,
                  new_tokens: int = 12) -> dict:
    ids = data.passkey_ids(tokenizer, n_tokens, depth, key, runner.device)
    result = runner.generate(ids, max_new_tokens=new_tokens)
    answer = tokenizer.decode(result.new_token_ids, skip_special_tokens=True)
    return {"prompt_tokens": ids.shape[1], "depth": depth, "key": key, "answer": answer.strip(),
            "correct": str(key) in answer, **{k: v for k, v in result.summary().items() if "cache" in k}}


def random_keys(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(10_000, 99_999) for _ in range(n)]
