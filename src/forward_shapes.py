"""
Task 5 - Tensor shapes in a single forward pass.

What it does
------------
Part A: runs ONE forward pass on the prompt with
        output_hidden_states=True and output_attentions=True,
        then prints the shape of every hidden state and attention matrix.
        It also checks two properties of the attention matrices:
          - each row sums to 1 (it is a softmax distribution),
          - everything above the diagonal is 0 (causal mask: a token cannot
            look at future tokens).
Part B: repeats the forward pass for several input lengths T and measures how
        big the attention matrices get, to show the T^2 growth.

Results -> results/forward_shapes.json ; write-up -> docs/shapes.md

Run:
    python src/forward_shapes.py
"""

import torch

from utils import load_config, load_model, load_tokenizer, resolve_device_and_dtype, save_json, set_seed


def part_a_shapes(model, tokenizer, prompt: str, device: str) -> dict:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    print(f"Prompt: {prompt!r}")
    print(f"input_ids shape: {tuple(input_ids.shape)}  (batch, seq_len)")
    print(f"tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}\n")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, output_attentions=True)

    # hidden_states: tuple with num_layers + 1 entries:
    #   [0]    = token embeddings (input to layer 1)
    #   [i]    = output of layer i
    #   [-1]   = output of the last layer AFTER the final RMSNorm (HF convention)
    # each has shape (batch, seq_len, hidden_size)
    print(f"len(hidden_states) = {len(out.hidden_states)}  (= num_layers + 1)")
    for i, h in enumerate(out.hidden_states):
        label = "embeddings" if i == 0 else f"after layer {i}"
        print(f"  hidden_states[{i:2d}] {label:<16} {tuple(h.shape)}")

    # attentions: tuple with num_layers entries, each of shape
    #   (batch, num_attention_heads, query_len, key_len)
    # With GQA the K/V heads are repeated to match the query heads before the
    # softmax, so we see num_attention_heads (32), not num_key_value_heads (8).
    print(f"\nlen(attentions) = {len(out.attentions)}  (= num_layers)")
    for i, a in enumerate(out.attentions):
        print(f"  attentions[{i:2d}] {tuple(a.shape)}")

    print(f"\nlogits shape: {tuple(out.logits.shape)}  (batch, seq_len, vocab_size)")

    # Sanity checks on the first layer's attention (in float32 for precision).
    a0 = out.attentions[0].float()
    row_sums = a0.sum(dim=-1)
    upper = torch.triu(a0, diagonal=1)
    print(f"\nCheck: rows sum to 1 -> max |sum-1| = {(row_sums - 1).abs().max().item():.2e}")
    print(f"Check: causal mask   -> max value above diagonal = {upper.abs().max().item():.2e}")

    return {
        "prompt": prompt,
        "input_ids_shape": list(input_ids.shape),
        "hidden_states_shapes": [list(h.shape) for h in out.hidden_states],
        "attentions_shapes": [list(a.shape) for a in out.attentions],
        "logits_shape": list(out.logits.shape),
    }


def part_b_growth(model, lengths: list[int], vocab_limit: int, device: str) -> list[dict]:
    """Measure attention-matrix size as a function of input length T."""
    print("\n" + "=" * 70 + "\nAttention size vs input length\n" + "=" * 70)
    print(f"{'T':>6} | {'shape per layer':>22} | {'MiB per layer':>13} | {'MiB all layers':>14}")
    rows = []
    for T in lengths:
        # Random (non-special) token IDs are fine here: we only care about shapes.
        ids = torch.randint(0, vocab_limit, (1, T), device=device)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True)
        a = out.attentions[0]
        per_layer = a.numel() * a.element_size()
        total = per_layer * len(out.attentions)
        rows.append({"T": T, "shape": list(a.shape), "bytes_per_layer": per_layer, "bytes_all_layers": total})
        print(f"{T:>6} | {str(tuple(a.shape)):>22} | {per_layer / 1024**2:>13.2f} | {total / 1024**2:>14.2f}")
        del out, a
        if device == "cuda":
            torch.cuda.empty_cache()

    # If size ~ T^2, doubling T multiplies size by 4.
    for prev, cur in zip(rows, rows[1:]):
        print(f"T x{cur['T'] / prev['T']:.0f} -> size x{cur['bytes_per_layer'] / prev['bytes_per_layer']:.0f}")
    return rows


def main() -> None:
    cfg = load_config()
    set_seed(cfg["seed"])
    device, dtype = resolve_device_and_dtype(cfg)
    model_id = cfg["models"]["small"]

    tokenizer = load_tokenizer(model_id)
    # "eager" attention is required: the fused SDPA/flash kernels never build
    # the full attention matrix, so they cannot return it.
    model = load_model(model_id, device, dtype, attn_implementation="eager")
    c = model.config
    print(f"{model_id}: layers={c.num_hidden_layers} hidden={c.hidden_size} "
          f"heads={c.num_attention_heads} kv_heads={c.num_key_value_heads} vocab={c.vocab_size}\n")

    shapes = part_a_shapes(model, tokenizer, cfg["prompt"], device)
    growth = part_b_growth(model, cfg["attention_growth_lengths"], tokenizer.vocab_size, device)

    save_json({"model_id": model_id, "dtype": str(dtype), **shapes, "attention_growth": growth},
              "forward_shapes.json")


if __name__ == "__main__":
    main()
