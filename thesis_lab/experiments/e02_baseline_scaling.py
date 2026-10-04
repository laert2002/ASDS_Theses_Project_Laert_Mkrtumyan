"""
E02 - Baseline scaling with context length (full KV cache, no compression).

Research questions
------------------
Q1  How do prefill time (TTFT), decode latency per token and peak memory grow
    with context length T? Do they follow the analytic model (E01)?
    Expected: KV memory linear; prefill time ~ linear + quadratic; decode
    latency ~ linear in T (it must read the whole cache every step).
Q2  How much memory do different attention BACKENDS need during prefill?
      eager      builds the (B, H, T, T) matrix -> quadratic memory
      sdpa       fused kernel -> linear memory, same FLOPs
      sdpa + all logits: shows the hidden cost of computing logits for all
                 T positions ((T x 128k vocab) matrix) when only the last is needed.
Q3  Where does the GPU run out of memory? (recorded as "OOM", not a crash)

Outputs: results/e02_scaling.csv, results/e02_backends.csv

Run:  python -m experiments.e02_baseline_scaling [--quick]
"""

from __future__ import annotations

import torch

from experiments.common import load_model, oom_safe, parse_args, setup, text_ids
from kvlab import measure, memory_model as mm, models
from kvlab.io import save_csv
from kvlab.runner import KVRunner


def prefill_only(model, ids, full_logits: bool) -> dict:
    measure.reset_peak(ids.device)
    with measure.Timer(ids.device) as t, torch.no_grad():
        model(input_ids=ids, use_cache=True, logits_to_keep=0 if full_logits else 1)
    return {"prefill_s": t.seconds, "peak_gib": measure.to_gib(measure.peak_bytes(ids.device))}


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    bc = cfg["baseline"]
    lengths = [512, 1024, 2048] if args.quick else bc["context_lengths"]
    steps = 8 if args.quick else bc["decode_steps"]

    model, tok = load_model(cfg, attn="sdpa")
    spec = mm.ModelSpec.from_hf_config(model.config)
    device = model.device
    weights = measure.allocated_bytes(device)
    print(f"weights on GPU: {measure.to_gib(weights):.2f} GiB")

    # Warm-up: the first CUDA calls include kernel selection / allocator growth.
    KVRunner(model).generate(text_ids(cfg, tok, "en", 128, device), max_new_tokens=4)

    rows = []
    for T in lengths:
        ids = text_ids(cfg, tok, "en", T, device)
        res = oom_safe(KVRunner(model).generate, ids, max_new_tokens=steps)
        # prefill caches T tokens; each of the (steps - 1) decode steps appends one more
        row = {"context": T, "analytic_kv_mib": measure.to_mib(mm.kv_cache_bytes(spec, T + steps - 1)),
               "weights_gib": measure.to_gib(weights)}
        row.update({"status": "OOM"} if res == "OOM" else {"status": "ok", **res.summary()})
        rows.append(row)
        print(row)
        measure.free_memory()
    save_csv(rows, "e02_scaling.csv")

    backend_rows = []
    for backend, full_logits in (("sdpa", False), ("sdpa", True), ("eager", False)):
        models.set_attention(model, backend)
        for T in lengths:
            if backend == "eager" and T > bc["eager_max_len"]:
                continue
            ids = text_ids(cfg, tok, "en", T, device)
            res = oom_safe(prefill_only, model, ids, full_logits)
            label = backend + ("+all_logits" if full_logits else "")
            backend_rows.append({"backend": label, "context": T,
                                 **({"status": "OOM"} if res == "OOM" else {"status": "ok", **res})})
            print(backend_rows[-1])
            measure.free_memory()
    models.set_attention(model, "sdpa")
    save_csv(backend_rows, "e02_backends.csv")


if __name__ == "__main__":
    main()
