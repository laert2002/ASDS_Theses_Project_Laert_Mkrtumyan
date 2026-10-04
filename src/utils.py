"""
Shared helpers used by every script in src/.

Why this file exists
--------------------
Every experiment needs the same four things: read the config, choose a device,
load a tokenizer/model, and save results. Keeping them here means that when you
later change *how* models are loaded (e.g. add quantization, a different
attention backend, or a KV-cache compression method for the thesis), you change
it in one place and every script picks it up.

All scripts are meant to be run from the repository root, e.g.
    python src/env_check.py
Python then puts src/ on sys.path, so `from utils import ...` works.
"""

import json
import random
from pathlib import Path

import torch
import yaml

# Repository root = parent of src/. Using absolute paths means scripts work
# no matter which directory you launch them from.
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "results"

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read the YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Make random tensors / sampling reproducible across runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device_and_dtype(cfg: dict) -> tuple[str, torch.dtype]:
    """
    Return (device, dtype) based on the config.

    float16 matmuls are slow or unsupported on many CPUs, so if CUDA is missing
    we fall back to float32 on CPU. This keeps scripts runnable (slowly) on a
    laptop for debugging, while real measurements should be done on GPU.
    """
    device = cfg.get("device", "cuda")
    dtype = _DTYPES[cfg.get("dtype", "float16")]
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -> using CPU + float32 (slow, for debugging only)")
        return "cpu", torch.float32
    return device, dtype


def load_tokenizer(model_id: str):
    """Load the (fast) tokenizer for a Hugging Face model id."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def load_model(model_id: str, device: str, dtype: torch.dtype, attn_implementation: str | None = None):
    """
    Load a causal LM directly onto `device` in `dtype` and put it in eval mode.

    attn_implementation:
        None     -> library default (usually "sdpa" = torch fused attention kernel)
        "eager"  -> plain PyTorch matmul + softmax. Slower, but it is the only
                    backend that returns attention matrices when
                    output_attentions=True (needed in Task 5).
    """
    from transformers import AutoModelForCausalLM

    kwargs = {"dtype": dtype, "device_map": device}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()  # disables dropout; we only do inference
    return model


def save_json(obj, filename: str) -> Path:
    """Save a JSON-serialisable object to results/<filename> and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[saved] {path.relative_to(ROOT)}")
    return path
