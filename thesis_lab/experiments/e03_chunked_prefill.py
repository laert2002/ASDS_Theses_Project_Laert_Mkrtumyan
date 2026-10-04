"""
E03 - Chunked prefill: trading prefill speed for peak memory.

Research questions
------------------
Q1  (full cache) Processing a T-token prompt in chunks of C tokens reduces the
    TEMPORARY activation memory (MLP intermediates, attention workspace grow
    with C, not T), but the kernels see smaller matrices -> lower GPU
    utilisation -> longer prefill. Where is the sweet spot?
Q2  (bounded memory, lab contribution) If an eviction policy runs BETWEEN
    chunks, the cache never exceeds budget + C tokens. Peak memory becomes
    INDEPENDENT of prompt length: a 128k-token prompt then fits in the same
    memory as an 8k one. The cost: quality (tokens are dropped) and a small
    overhead per chunk. Compare with the full cache, which hits OOM.

Outputs: results/e03_chunk_sizes.csv, results/e03_bounded.csv

Run:  python -m experiments.e03_chunked_prefill [--quick]
"""

from __future__ import annotations

import torch

from experiments.common import load_model, oom_safe, parse_args, setup, text_ids
from kvlab import compat, measure, policies
from kvlab.io import save_csv
from kvlab.runner import KVRunner


def timed_prefill(runner: KVRunner, ids: torch.Tensor) -> dict:
    runner.reset()
    measure.reset_peak(runner.device)
    with measure.Timer(runner.device) as t:
        runner.prefill(ids)
    return {"prefill_s": t.seconds, "peak_gib": measure.to_gib(measure.peak_bytes(runner.device)),
            "cache_mib": measure.to_mib(compat.cache_bytes(runner.cache))}


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    cc = cfg["chunked_prefill"]
    model, tok = load_model(cfg, attn="sdpa")
    device = model.device

    T = 4096 if args.quick else cc["context_length"]
    chunks = [None, 1024, 256] if args.quick else cc["chunk_sizes"]
    ids = text_ids(cfg, tok, "en", T, device)
    timed_prefill(KVRunner(model), ids[:, :256])  # warm-up

    rows = []
    for c in chunks:
        res = oom_safe(timed_prefill, KVRunner(model, prefill_chunk=c), ids)
        rows.append({"context": T, "chunk": c or T, **({"status": "OOM"} if res == "OOM" else {"status": "ok", **res})})
        print(rows[-1])
        measure.free_memory()
    save_csv(rows, "e03_chunk_sizes.csv")

    b = cc["bounded"]
    lengths = [2048, 4096, 8192] if args.quick else b["context_lengths"]
    spec = cfg["policies"][b["policy"]]
    rows = []
    for T in lengths:
        ids = text_ids(cfg, tok, "en", T, device)
        for label, runner in (
            ("full", KVRunner(model)),
            (f"{b['policy']}@{b['budget']}+chunk{b['chunk']}",
             KVRunner(model, policies.build(spec, b["budget"]), prefill_chunk=b["chunk"])),
        ):
            res = oom_safe(timed_prefill, runner, ids)
            rows.append({"context": T, "method": label,
                         **({"status": "OOM"} if res == "OOM" else {"status": "ok", **res})})
            print(rows[-1])
            measure.free_memory()
    save_csv(rows, "e03_bounded.csv")


if __name__ == "__main__":
    main()
