"""
Inputs for long-context experiments.

1. Natural long text (for perplexity and realistic timing)
   Wikipedia via the `datasets` library in streaming mode (nothing huge is
   downloaded), one language at a time: "en" English, "hy" Armenian, ...
   The text is cached in thesis_lab/data_cache/ so later runs work offline.

2. Passkey retrieval (Mohtashami & Jaggi, 2023, "Landmark Attention")
   A number hidden at a chosen DEPTH inside repetitive filler text; the model
   must repeat it at the end. Directly tests whether a compressed cache still
   contains a specific old token. Works with base (non-instruct) models because
   the prompt ends with "The pass key is" and the model only has to continue it.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .io import LAB_ROOT

DATA_CACHE = LAB_ROOT / "data_cache"


def load_text(lang: str = "en", min_chars: int = 400_000, source: str = "wikipedia",
              file: str | None = None) -> str:
    """Return at least `min_chars` characters of natural text in `lang`."""
    if source == "file":
        return Path(file).read_text(encoding="utf-8")

    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    cached = DATA_CACHE / f"wikipedia_{lang}.txt"
    if cached.exists() and cached.stat().st_size >= min_chars:
        return cached.read_text(encoding="utf-8")

    from datasets import load_dataset

    stream = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    parts, n = [], 0
    for row in stream:
        if len(row["text"]) < 3000:  # skip stubs (many bot-generated in small wikis)
            continue
        parts.append(row["text"])
        n += len(row["text"])
        if n >= min_chars:
            break
    text = "\n\n".join(parts)
    cached.write_text(text, encoding="utf-8")
    return text


def prompt_ids(tokenizer, text: str, n_tokens: int, device) -> torch.Tensor:
    """First n_tokens tokens of `text` (BOS included), shape (1, n_tokens)."""
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[1] < n_tokens:
        raise ValueError(f"text has only {ids.shape[1]} tokens, need {n_tokens}: increase min_chars")
    return ids[:, :n_tokens].to(device)


INTRO = ("There is an important info hidden inside a lot of irrelevant text. "
         "Find it and memorize it. I will quiz you about the important information there.\n")
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
NEEDLE = "\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
QUESTION = "\nWhat is the pass key? The pass key is"


def passkey_ids(tokenizer, n_tokens: int, depth: float, key: int, device) -> torch.Tensor:
    """Prompt of ~n_tokens tokens with the passkey at relative `depth` (0 = start, 1 = end)."""
    def count(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False).input_ids)

    fixed = 1 + count(INTRO) + count(NEEDLE.format(key=key)) + count(QUESTION)
    n_rep = max(1, (n_tokens - fixed) // count(" " + FILLER))
    before = round(depth * n_rep)
    text = (INTRO + " ".join([FILLER] * before) + NEEDLE.format(key=key)
            + " ".join([FILLER] * (n_rep - before)) + QUESTION)
    return tokenizer(text, return_tensors="pt").input_ids.to(device)
