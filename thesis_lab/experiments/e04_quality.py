"""
E04 - Quality vs memory for every KV-cache policy, in English and Armenian.

Research questions
------------------
Q1  For the same MEMORY, which policy loses the least quality?
    Policies are compared on the x-axis "logical KV bytes" (not "budget"),
    because 1024 fp16 tokens and 4096 int4 tokens cost roughly the same memory.
    This equal-memory comparison is the core trade-off plot of the thesis.
Q2  Does eviction hurt more far from the beginning? (ppl_second_half)
Q3  (lab contribution) Is KV compression more harmful for a low-resource
    language? Armenian needs more tokens for the same content, so a fixed
    token budget covers LESS text. Compare with bits-per-byte (BPB), which is
    tokenizer-independent, and with "KV bytes per byte of text".
Q4  Do the lab variants (h2o_mean, h2o_adaptive, tiered*) beat their baselines?

Method: every text is fed in chunks of `eval_chunk` tokens; the policy
compresses the cache between chunks (see KVRunner.nll). Results are appended
after every run, so an interrupted sweep keeps its finished rows.

Outputs: results/e04_quality.csv

Run:  python -m experiments.e04_quality [--quick]
"""

from __future__ import annotations

from experiments.common import load_model, parse_args, setup, text_ids
from kvlab import measure, policies, quality, scoring
from kvlab.io import save_csv
from kvlab.runner import KVRunner


def main() -> None:
    args = parse_args(__doc__)
    cfg = setup(args)
    qc = cfg["quality"]
    eval_tokens = 1024 if args.quick else qc["eval_tokens"]
    budgets = [128, 256] if args.quick else qc["budgets"]
    names = ["full", "streaming_llm", "h2o", "kivi4", "tiered4"] if args.quick else qc["policies"]

    model, tok = load_model(cfg, attn=scoring.ATTN_NAME)  # needed by score-based policies
    rows = []
    for lang in cfg["data"]["languages"]:
        ids = text_ids(cfg, tok, lang, eval_tokens, model.device)
        for name in names:
            spec = cfg["policies"][name]
            cls = policies.REGISTRY[spec["type"]]
            for budget in (budgets if cls.uses_budget else [None]):
                policy = policies.build(spec, budget)
                runner = KVRunner(model, policy, score_window=cfg["score_window"])
                with measure.Timer(model.device) as t:
                    res = quality.perplexity(runner, ids, chunk=qc["eval_chunk"], tokenizer=tok)
                row = {"lang": lang, "policy": name, "budget": budget, "seconds": t.seconds, **res,
                       "kv_bytes_per_context_token": res["cache_logical_bytes"] / eval_tokens,
                       "kv_bytes_per_text_byte": res["cache_logical_bytes"] / res["text_bytes"],
                       "policy_config": policy.config()}
                rows.append(row)
                print(f"{lang} {name:>14} budget={budget!s:>5}  ppl={res['ppl']:8.3f}  bpb={res['bpb']:.4f}  "
                      f"kv={res['cache_logical_bytes'] / 2**20:7.1f} MiB")
                save_csv(rows, "e04_quality.csv")
                measure.free_memory()


if __name__ == "__main__":
    main()
