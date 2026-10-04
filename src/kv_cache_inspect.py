"""
Task 7 - Inspecting the KV cache.

What it does
------------
1. Runs the prompt through the model with use_cache=True (the "prefill" step).
2. Inspects out.past_key_values: how many entries it has, and the shape,
   dtype and memory of every key/value tensor.
3. Checks every shape against the model config (layers, kv heads, head_dim).
4. Runs ONE decoding step (feeds only the next token + the cache) and shows
   that the cache grows by exactly one position along the sequence axis.
5. Compares measured memory with the formula
       2 * layers * kv_heads * head_dim * seq_len * bytes_per_element.

Results -> results/kv_cache.json ; write-up -> docs/kv_cache_structure.md

Run:
    python src/kv_cache_inspect.py
"""

import torch

from utils import load_config, load_model, load_tokenizer, resolve_device_and_dtype, save_json, set_seed


def iter_kv(cache):
    """
    Yield (key, value) per layer, independent of the transformers version.

    The cache object changed across versions:
      * transformers >= 4.56 : DynamicCache with .layers[i].keys / .values
      * 4.36 - 4.55          : DynamicCache with .key_cache / .value_cache lists
      * older                : tuple of (key, value) tuples ("legacy" format)
    """
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            yield layer.keys, layer.values
    elif hasattr(cache, "key_cache"):
        yield from zip(cache.key_cache, cache.value_cache)
    else:
        for k, v in cache:
            yield k, v


def describe(cache, title: str) -> list[dict]:
    kv = list(iter_kv(cache))
    print(f"\n{title}")
    print(f"  cache type: {type(cache).__name__} | entries (layers): {len(kv)}")
    rows = []
    for i, (k, v) in enumerate(kv):
        rows.append({"layer": i, "key_shape": list(k.shape), "value_shape": list(v.shape),
                     "bytes": k.numel() * k.element_size() + v.numel() * v.element_size()})
        print(f"  layer {i:2d}: key {tuple(k.shape)}  value {tuple(v.shape)}  {k.dtype}")
    return rows


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    device, dtype = resolve_device_and_dtype(cfg)
    model_id = cfg["models"]["small"]

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, device, dtype)
    c = model.config
    L, Hkv = c.num_hidden_layers, c.num_key_value_heads
    D = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    print(f"{model_id}: layers={L} heads={c.num_attention_heads} kv_heads={Hkv} head_dim={D}")

    inputs = tokenizer(cfg["prompt"], return_tensors="pt").to(device)
    T = inputs["input_ids"].shape[1]
    print(f"Prompt {cfg['prompt']!r} -> T = {T} tokens (incl. BOS)")

    # ---- Prefill: process the whole prompt at once, fill the cache ----------
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    cache = out.past_key_values
    prefill = describe(cache, "After prefill:")

    # Every tensor must be (batch, num_key_value_heads, seq_len, head_dim).
    expected = [inputs["input_ids"].shape[0], Hkv, T, D]
    assert len(prefill) == L, f"expected {L} layers, got {len(prefill)}"
    assert all(r["key_shape"] == expected and r["value_shape"] == expected for r in prefill), \
        f"unexpected shapes, expected {expected}"
    print(f"  OK: {L} entries, each K and V has shape {tuple(expected)} = (batch, kv_heads, seq_len, head_dim)")

    bytes_per_elem = torch.finfo(model.dtype).bits // 8
    measured = sum(r["bytes"] for r in prefill)
    formula = 2 * L * Hkv * D * T * bytes_per_elem
    print(f"  memory: measured {measured:,} B | formula 2*{L}*{Hkv}*{D}*{T}*{bytes_per_elem} = {formula:,} B")

    # ---- One decode step: feed only the new token, reuse the cache ----------
    # The model only computes Q/K/V for the new token and attends over the
    # cached K/V, instead of re-processing all T+1 tokens.
    next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)
    print(f"\nGreedy next token: {tokenizer.decode(next_id[0])!r}")
    with torch.no_grad():
        out2 = model(input_ids=next_id, past_key_values=cache, use_cache=True)
    decode = describe(out2.past_key_values, "After 1 decode step:")
    print(f"  seq_len grew {prefill[0]['key_shape'][2]} -> {decode[0]['key_shape'][2]}")

    per_token = 2 * L * Hkv * D * bytes_per_elem
    print(f"\nKV cache per token: {per_token:,} B = {per_token / 1024:.0f} KiB")
    for n in (1_024, 32_768, 131_072):
        print(f"  {n:>7,} tokens -> {per_token * n / 1024**3:6.2f} GiB")

    save_json({"model_id": model_id, "dtype": str(model.dtype), "prompt_tokens": T,
               "layers": L, "kv_heads": Hkv, "head_dim": D,
               "prefill": prefill, "after_one_decode_step": decode,
               "bytes_measured_prefill": measured, "bytes_formula_prefill": formula,
               "bytes_per_token": per_token}, "kv_cache.json")


if __name__ == "__main__":
    main()
