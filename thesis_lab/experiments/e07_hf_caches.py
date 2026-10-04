"""
E07 - Hugging Face built-in cache implementations (REAL memory savings).

The lab's quantization policies are simulated (exact quality, analytic memory).
This experiment measures the production implementations shipped with
transformers, through the standard generate() API:

    dynamic      default; grows with torch.cat (copies the cache every step)
    static       pre-allocated to the maximum length: no re-allocation, allows
                 torch.compile / CUDA graphs; wastes memory for short outputs
    offloaded    keeps only the current layer's cache on the GPU, the rest in
                 CPU RAM, prefetching the next layer. Trades PCIe transfer time
                 for GPU memory (memory <-> latency trade-off)
    quantized_*  real int4 storage via optimum-quanto or HQQ; values are
                 de-quantized on the fly (memory <-> compute trade-off)

Unavailable implementations (missing optional package, unsupported by your
transformers version) are recorded with their error message instead of crashing.
Optional packages:  pip install optimum-quanto   /   pip install hqq

Outputs: results/e07_hf_caches.csv

Run:  python -m experiments.e07_hf_caches [--quick]
"""

from __future__ import annotations

import torch

from experiments.common import load_model, parse_args, setup, text_ids
from kvlab import measure
from kvlab.io import save_csv


def generate_kwargs(impl: str) -> dict:
    if impl.startswith("quantized_"):
        _, backend, bits = impl.split("_")
        return {"cache_implementation": "quantized", "cache_config": {"backend": backend, "nbits": int(bits)}}
    return {"cache_implementation": impl}


def run(model, ids, new_tokens: int, impl: str) -> dict:
    device = ids.device
    measure.reset_peak(device)
    with measure.Timer(device) as t, torch.no_grad():
        out = model.generate(ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False,
                             pad_token_id=model.config.eos_token_id, **generate_kwargs(impl))
    produced = out.shape[1] - ids.shape[1]
    return {"total_s": t.seconds, "tokens_per_s": produced / t.seconds,
            "peak_gib": measure.to_gib(measure.peak_bytes(device)), "new_tokens": produced}


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    hc = cfg["hf_caches"]
    lengths = [1024] if args.quick else hc["context_lengths"]
    new_tokens = 16 if args.quick else hc["new_tokens"]

    model, tok = load_model(cfg, attn="sdpa")
    run(model, text_ids(cfg, tok, "en", 64, model.device), 4, "dynamic")  # warm-up

    rows = []
    for T in lengths:
        ids = text_ids(cfg, tok, "en", T, model.device)
        for impl in hc["implementations"]:
            row = {"context": T, "implementation": impl}
            try:
                row.update(status="ok", **run(model, ids, new_tokens, impl))
            except torch.cuda.OutOfMemoryError:
                row["status"] = "OOM"
            except (ImportError, ValueError, RuntimeError, TypeError, NotImplementedError) as e:
                row.update(status="unavailable", error=f"{type(e).__name__}: {str(e)[:200]}")
            rows.append(row)
            print(row)
            measure.free_memory()
    save_csv(rows, "e07_hf_caches.csv")


if __name__ == "__main__":
    main()
