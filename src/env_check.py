"""
Task 1 - Environment check.

What it does
------------
1. Prints library versions and GPU information, so you know the environment
   is correct before running anything expensive.
2. Loads Llama-3.2-1B in float16 on the GPU.
3. Generates 20 new tokens from "The capital of France is" and prints them.

Run:
    python src/env_check.py

If this script works, then PyTorch+CUDA, transformers, accelerate, and your
Hugging Face access to the gated Llama models are all set up correctly.
"""

import time

import torch
import transformers

from utils import load_config, load_model, load_tokenizer, resolve_device_and_dtype, set_seed


def print_environment() -> None:
    print("=" * 60)
    print(f"torch          : {torch.__version__}")
    print(f"transformers   : {transformers.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA version   : {torch.version.cuda}")
        print(f"GPU            : {props.name}")
        print(f"GPU memory     : {props.total_memory / 1024**3:.1f} GiB")
    print("=" * 60)


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    print_environment()

    model_id = cfg["models"]["small"]
    device, dtype = resolve_device_and_dtype(cfg)

    t0 = time.perf_counter()
    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, device, dtype)
    print(f"Loaded {model_id} in {time.perf_counter() - t0:.1f}s | dtype={model.dtype} | device={model.device}")

    # Tokenize the prompt. return_tensors="pt" gives PyTorch tensors of shape
    # (batch=1, seq_len). The Llama tokenizer automatically prepends the
    # <|begin_of_text|> (BOS) token.
    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(device)

    # Greedy decoding (do_sample=False) -> deterministic output, so results are
    # reproducible. Llama has no pad token, so we reuse EOS to silence a warning.
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=cfg["max_new_tokens"],
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # generate() returns prompt + new tokens; slice to separate them.
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[0, prompt_len:]
    print(f"\nPrompt      : {cfg['prompt']!r}")
    print(f"Generated   : {tokenizer.decode(new_ids, skip_special_tokens=True)!r}")
    print(f"Full output : {tokenizer.decode(output_ids[0], skip_special_tokens=True)!r}")
    print(f"New tokens  : {len(new_ids)}")

    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
