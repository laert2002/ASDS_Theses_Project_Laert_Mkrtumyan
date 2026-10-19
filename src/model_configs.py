"""
Task 4 - Model configuration comparison.

What it does
------------
Downloads ONLY config.json (a few KB, no weights) for both models, prints the
full config, extracts the architecture numbers that matter for memory/compute,
and prints a markdown comparison table.

Results -> results/model_configs.json ; write-up -> docs/model_configs.md

Run:
    python src/model_configs.py
"""

from transformers import AutoConfig

from utils import load_config, save_json

BYTES_PER_ELEMENT = {"float16": 2, "bfloat16": 2, "float32": 4}


def load_model_config(model_id: str):
    """Load a model config, skipping gated repos with a clear warning."""
    try:
        return AutoConfig.from_pretrained(model_id)
    except Exception as exc:  # pragma: no cover - exercised by integration tests
        msg = str(exc).lower()
        if (
            "gated" in msg
            or "restricted" in msg
            or "authorized" in msg
            or "unauthorized" in msg
            or "401" in msg
            or "403" in msg
        ):
            print(f"[warn] Skipping {model_id}: access is restricted or the repo is gated.")
            return None
        raise


def summarize(config, bytes_per_elem: int) -> dict:
    # Some configs (e.g. Llama-3.1-8B) don't store head_dim explicitly. Then it
    # is defined as hidden_size / num_attention_heads.
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv = config.num_key_value_heads

    # KV cache per token: for every layer we store one Key and one Value vector
    # per KV head, each of length head_dim.
    #   bytes/token = 2 (K and V) * layers * n_kv_heads * head_dim * bytes_per_elem
    kv_per_token = 2 * layers * n_kv * head_dim * bytes_per_elem
    # Same thing if the model used plain multi-head attention (n_kv == n_heads).
    kv_per_token_mha = 2 * layers * n_heads * head_dim * bytes_per_elem

    return {
        "num_hidden_layers": layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "gqa_group_size": n_heads // n_kv,  # query heads sharing one KV head
        "kv_cache_bytes_per_token": kv_per_token,
        "kv_cache_bytes_per_token_if_mha": kv_per_token_mha,
    }


def main() -> None:
    cfg = load_config()
    bpe = BYTES_PER_ELEMENT[cfg["dtype"]]

    summaries = {}
    for name, model_id in cfg["models"].items():
        config = load_model_config(model_id)
        if config is None:
            continue
        print("=" * 70, f"\n{model_id}  (full model.config)\n" + "=" * 70)
        print(config)  # the same object you get as model.config after loading
        summaries[model_id] = summarize(config, bpe)

    if not summaries:
        raise RuntimeError(
            "No model configs could be loaded. Grant access to the gated repo(s) or update the entries in configs/default.yaml."
        )

    ids = list(summaries)
    print("\nCOMPARISON (paste into docs/model_configs.md)\n")
    print("| Field | " + " | ".join(i.split("/")[-1] for i in ids) + " |")
    print("|---|" + "---|" * len(ids))
    for field in summaries[ids[0]]:
        print(f"| {field} | " + " | ".join(f"{summaries[i][field]:,}" for i in ids) + " |")

    # A long-context example makes the numbers concrete.
    ctx = 128 * 1024
    print(f"\nKV cache for one sequence of {ctx:,} tokens ({cfg['dtype']}):")
    for i in ids:
        s = summaries[i]
        print(f"  {i.split('/')[-1]:>14}: GQA {s['kv_cache_bytes_per_token'] * ctx / 1024**3:6.1f} GiB"
              f" | if MHA {s['kv_cache_bytes_per_token_if_mha'] * ctx / 1024**3:6.1f} GiB")

    save_json({"dtype": cfg["dtype"], "models": summaries}, "model_configs.json")


if __name__ == "__main__":
    main()
