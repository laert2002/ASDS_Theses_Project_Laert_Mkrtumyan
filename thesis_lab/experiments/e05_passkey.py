"""
E05 - Passkey retrieval: does the compressed cache still contain a specific old fact?

Research questions
------------------
Q1  Perplexity averages over all tokens and can hide catastrophic failures.
    Retrieval asks for ONE token sequence far in the past - exactly what
    eviction can delete. Which policies keep retrieval working, and at which
    depth (position of the needle) do they fail?
Q2  Eviction is irreversible; mixed precision (tiered) keeps a blurred copy.
    Does "demote instead of evict" rescue retrieval at equal memory?
Q3  How does accuracy change with context length at a fixed budget?

Outputs: results/e05_passkey_raw.csv (every trial), results/e05_passkey.csv (accuracy)

Run:  python -m experiments.e05_passkey [--quick]
"""

from __future__ import annotations

from collections import defaultdict

from experiments.common import load_model, oom_safe, parse_args, setup
from kvlab import measure, policies, quality, scoring
from kvlab.io import save_csv
from kvlab.runner import KVRunner


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    pc = cfg["passkey"]
    lengths = [1024, 2048] if args.quick else pc["context_lengths"]
    depths = [0.2, 0.8] if args.quick else pc["depths"]
    trials = 2 if args.quick else pc["trials"]
    names = ["full", "streaming_llm", "h2o", "tiered2"] if args.quick else pc["policies"]

    model, tok = load_model(cfg, attn=scoring.ATTN_NAME)
    keys = quality.random_keys(trials, cfg["seed"])
    raw = []
    for name in names:
        spec = cfg["policies"][name]
        budget = pc["budget"] if policies.REGISTRY[spec["type"]].uses_budget else None
        runner = KVRunner(model, policies.build(spec, budget), score_window=cfg["score_window"],
                          prefill_chunk=pc["prefill_chunk"])
        for T in lengths:
            for depth in depths:
                for key in keys:
                    res = oom_safe(quality.passkey_trial, runner, tok, T, depth, key)
                    row = {"policy": name, "budget": budget, "context": T}
                    row.update({"depth": depth, "key": key, "correct": None, "status": "OOM"} if res == "OOM"
                               else {**res, "status": "ok"})
                    raw.append(row)
                    measure.free_memory()
                print(f"{name:>14} T={T:>6} depth={depth:.1f} "
                      f"acc={sum(r['correct'] or 0 for r in raw[-trials:]) / trials:.2f}")
        save_csv(raw, "e05_passkey_raw.csv")

    acc = defaultdict(list)
    for r in raw:
        if r["status"] == "ok":
            acc[(r["policy"], r["context"], r["depth"])].append(r["correct"])
    save_csv([{"policy": p, "context": T, "depth": d, "accuracy": sum(v) / len(v), "trials": len(v)}
              for (p, T, d), v in acc.items()], "e05_passkey.csv")


if __name__ == "__main__":
    main()
