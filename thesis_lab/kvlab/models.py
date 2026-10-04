"""
Model/tokenizer loading for the lab.

attn_implementation choices:
    "kvlab_sdpa" (default) - fused SDPA + optional attention-score recording.
                             Required by kvlab.runner.KVRunner.
    "sdpa"                 - stock PyTorch fused attention (baselines, HF caches)
    "eager"                - materialises attention matrices (O(T^2) memory)
    "flash_attention_2"    - needs the flash-attn package and an Ampere+ GPU
"""

from __future__ import annotations

import torch

from . import scoring

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def resolve_device(device: str) -> str:
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available -> CPU (debugging only, timings meaningless)")
        return "cpu"
    return device


def load(model_id: str, dtype: str = "float16", device: str = "cuda", attn: str = scoring.ATTN_NAME):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(device)
    torch_dtype = DTYPES[dtype] if device != "cpu" else torch.float32
    if attn == scoring.ATTN_NAME:
        scoring.register()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch_dtype, device_map=device, attn_implementation=attn
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def set_attention(model, attn: str) -> None:
    """Switch the attention backend of a loaded model (avoids reloading the weights)."""
    if attn == scoring.ATTN_NAME:
        scoring.register()
    if hasattr(model, "set_attn_implementation"):  # transformers >= 4.56
        model.set_attn_implementation(attn)
    else:  # attention layers read this field at every forward pass
        model.config._attn_implementation = attn
