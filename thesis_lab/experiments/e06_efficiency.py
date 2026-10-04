"""
E06 - Runtime cost of the policies: TTFT, decode latency, peak memory.

Research questions
------------------
Q1  Compression is not free: scoring, top-k and gathering add compute to
    every step. Does the smaller cache make decoding FASTER (less memory to
    read, decode is memory-bound, see E01 roofline) or SLOWER (overhead)?
    Answer depends on context length -> measure at several T.
Q2  Peak memory during prefill vs decode for each policy.
Q3  The compute extreme: NO cache. Each step re-runs the whole sequence.
    KV memory is zero but per-token latency grows linearly with T (and total
    compute quadratically). Where does it become unusable?

Note: quantization policies here are SIMULATED (see kvlab/policies/quant.py),
so their measured memory/latency reflects fp16 storage plus quantization
overhead. Their "logical" memory column shows what a packed implementation needs.

Outputs: results/e06_efficiency.csv, results/e06_no_cache.csv

Run:  python -m experiments.e06_efficiency [--quick]
"""

from __future__ import annotations

from experiments.common import load_model, oom_safe, parse_args, setup, text_ids
from kvlab import measure, policies, scoring
from kvlab.io import save_csv
from kvlab.runner import KVRunner, generate_without_cache


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    ec = cfg["efficiency"]
    lengths = [2048, 4096] if args.quick else ec["context_lengths"]
    new_tokens = 16 if args.quick else ec["new_tokens"]
    names = ["full", "streaming_llm", "h2o", "kivi4"] if args.quick else ec["policies"]

    model, tok = load_model(cfg, attn=scoring.ATTN_NAME)
    device = model.device
    KVRunner(model).generate(text_ids(cfg, tok, "en", 128, device), max_new_tokens=4)  # warm-up

    rows = []
    for T in lengths:
        ids = text_ids(cfg, tok, "en", T, device)
        for name in names:
            spec = cfg["policies"][name]
            budget = ec["budget"] if policies.REGISTRY[spec["type"]].uses_budget else None
            runner = KVRunner(model, policies.build(spec, budget), score_window=cfg["score_window"],
                              prefill_chunk=ec["prefill_chunk"])
            res = oom_safe(runner.generate, ids, max_new_tokens=new_tokens)
            rows.append({"context": T, "policy": name, "budget": budget,
                         **({"status": "OOM"} if res == "OOM" else {"status": "ok", **res.summary()})})
            print(rows[-1])
            measure.free_memory()
        save_csv(rows, "e06_efficiency.csv")

    nc = ec["no_cache"]
    rows = []
    for T in ([256, 512] if args.quick else nc["context_lengths"]):
        ids = text_ids(cfg, tok, "en", T, device)
        n = 4 if args.quick else nc["new_tokens"]
        cached = oom_safe(KVRunner(model).generate, ids, max_new_tokens=n)
        uncached = oom_safe(generate_without_cache, model, ids, n)
        rows.append({
            "context": T, "new_tokens": n,
            "cached_decode_ms": None if cached == "OOM" else cached.summary()["decode_ms_mean"],
            "no_cache_step_ms": None if uncached == "OOM" else 1e3 * sum(uncached["step_s"]) / n,
            "no_cache_peak_gib": None if uncached == "OOM" else measure.to_gib(uncached["peak_bytes"]),
            "cached_peak_gib": None if cached == "OOM" else cached.summary()["peak_decode_gib"],
        })
        print(rows[-1])
        measure.free_memory()
    save_csv(rows, "e06_no_cache.csv")


if __name__ == "__main__":
    main()
