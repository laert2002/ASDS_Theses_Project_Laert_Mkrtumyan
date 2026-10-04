"""
E01 - Analytic memory/compute study (no GPU, no model download required).

Research questions
------------------
Q1  How big is the KV cache compared to the weights, and at which context
    length does it become the dominant memory cost?
Q2  How much do GQA/MQA and KV quantization shrink it (incl. scale overhead)?
Q3  What is the longest context that fits on common GPUs, for batch 1 and 8?
Q4  Is decoding compute-bound or memory-bound, and how does the answer change
    with context length and batch size? (roofline)
Q5  At which prompt length does quadratic attention dominate prefill FLOPs?
Q6  What does it cost to drop the cache entirely (recompute everything)?

Outputs: results/e01_*.csv and e01_summary.json (plots: experiments/plots.py)

Run:  python -m experiments.e01_analytic_memory
"""

from __future__ import annotations

from experiments.common import parse_args, setup
from kvlab import memory_model as mm
from kvlab.io import save_csv, save_json

CONTEXTS = [1024, 4096, 16384, 32768, 65536, 131072]
KV_FORMATS = {"fp16": (16, None), "int8_g32": (8, 32), "int4_g32": (4, 32), "int2_g32": (2, 32)}


def load_specs(cfg) -> list[mm.ModelSpec]:
    """Prefer the real configs from the Hub; fall back to the built-in presets offline."""
    try:
        from transformers import AutoConfig

        return [mm.ModelSpec.from_hf_config(AutoConfig.from_pretrained(m), m.split("/")[-1])
                for m in (cfg["model"], cfg["model_large"])]
    except (ImportError, OSError) as e:
        print(f"[info] using built-in presets ({type(e).__name__})")
        return list(mm.PRESETS.values())


def main() -> None:
    cfg = setup(parse_args(__doc__))
    specs = load_specs(cfg)
    summary = {}

    # Q1 + Q2: per-model overview
    overview = []
    for m in specs:
        w = mm.weight_bytes(m)
        row = {"model": m.name, "params_B": m.total_params / 1e9, "weights_GiB": w / mm.GiB}
        for fmt, (bits, g) in KV_FORMATS.items():
            row[f"kv_KiB_per_token_{fmt}"] = mm.kv_bytes_per_token(m, bits, g) / 1024
        row["tokens_where_kv_equals_weights_fp16"] = int(w / mm.kv_bytes_per_token(m))
        overview.append(row)
    save_csv(overview, "e01_overview.csv")
    summary["overview"] = overview

    # Q2: attention variants (what-if the same model used MHA or MQA)
    variants = []
    for m in specs:
        for label, h in (("MHA", m.heads), ("GQA (actual)", m.kv_heads), ("MQA", 1)):
            v = m.with_kv_heads(h)
            variants.append({"model": m.name, "variant": label, "kv_heads": h,
                             "kv_KiB_per_token": mm.kv_bytes_per_token(v) / 1024,
                             "kv_GiB_at_128k": mm.kv_cache_bytes(v, 131072) / mm.GiB})
    save_csv(variants, "e01_attention_variants.csv")

    # KV cache and other O(T)/O(T^2) buffers vs context length
    rows = []
    for m in specs:
        for T in CONTEXTS:
            row = {"model": m.name, "context": T}
            for fmt, (bits, g) in KV_FORMATS.items():
                row[f"kv_GiB_{fmt}"] = mm.kv_cache_bytes(m, T, kv_bits=bits, group_size=g) / mm.GiB
            row["kv_GiB_window_2048"] = mm.kv_cache_bytes(m, T, budget=2048) / mm.GiB
            row["eager_attn_one_layer_GiB"] = mm.eager_attention_bytes(m, T) / mm.GiB
            row["prefill_logits_all_GiB"] = mm.prefill_logits_bytes(m, T) / mm.GiB
            rows.append(row)
    save_csv(rows, "e01_kv_vs_context.csv")

    # Q3: longest context that fits
    fits = []
    for m in specs:
        for gpu in mm.GPUS.values():
            for batch in (1, 8):
                for fmt, (bits, g) in KV_FORMATS.items():
                    fits.append({"model": m.name, "gpu": gpu.name, "batch": batch, "kv_format": fmt,
                                 "max_context": mm.max_context(m, gpu.memory_gib, batch, kv_bits=bits, group_size=g)})
    save_csv(fits, "e01_max_context.csv")

    # Q4: roofline of one decode step
    roof = []
    gpu = mm.GPUS["A100-80GB"]
    for m in specs:
        for batch in (1, 16):
            for T in CONTEXTS:
                r = mm.decode_step_roofline(m, gpu, T, batch)
                roof.append({"model": m.name, "gpu": gpu.name, "batch": batch, "context": T, **r,
                             "kv_share_of_bytes": mm.kv_cache_bytes(m, T, batch) / r["bytes"]})
    save_csv(roof, "e01_decode_roofline.csv")

    # Q5: prefill FLOPs split
    split = []
    for m in specs:
        crossover = m.matmul_params / (m.layers * m.heads * m.head_dim)  # 2NT = 2*L*H*Dh*T^2
        summary[f"{m.name}_attention_equals_linear_flops_at_T"] = int(crossover)
        for T in CONTEXTS:
            f = mm.prefill_flops(m, T)
            split.append({"model": m.name, "context": T, "TFLOPs": f["total"] / 1e12,
                          "attention_share": f["attention"] / f["total"]})
    save_csv(split, "e01_prefill_flops.csv")

    # Q6: recompute instead of caching
    rec = []
    for m in specs:
        for prompt in (512, 2048, 8192):
            rec.append({"model": m.name, "prompt": prompt, "new_tokens": 256,
                        **mm.recompute_vs_cache(m, prompt, 256)})
    save_csv(rec, "e01_recompute_vs_cache.csv")

    save_json(summary, "e01_summary.json")
    for row in overview:
        print(f"{row['model']:>14}: {row['params_B']:.2f}B params, weights {row['weights_GiB']:.1f} GiB, "
              f"KV {row['kv_KiB_per_token_fp16']:.0f} KiB/token, KV = weights at "
              f"{row['tokens_where_kv_equals_weights_fp16']:,} tokens")


if __name__ == "__main__":
    main()
