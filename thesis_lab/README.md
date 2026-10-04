# thesis_lab — Memory–Computation Trade-offs in Long-Context LLM Inference

A self-contained research workspace for the thesis. It is **separate from `src/`**: the foundation scripts (Tasks 1–7) stay untouched, and you integrate pieces from here when ready.

> **Status: written, not yet executed.** None of this code has been run: there was no GPU and no `transformers` on the authoring machine. Do step 1 of *Getting started* (run the tests) before trusting any number. The tests are designed to catch the bugs that would silently corrupt results.

---

## 1. The idea in one picture

```
               memory ▲
                      │  full KV cache ──────────────── grows O(T), OOM at long T
                      │
   quantize (KIVI) ───┤  same shape, 3–8× smaller slope
   tiered precision ──┤  small full-precision core + low-bit rest
                      │
   evict (H2O, ...) ──┤  flat line: O(budget), independent of T
                      │
   no cache at all ───┤  zero KV memory, but compute per token grows O(T)
                      └────────────────────────────────────────────▶ context length T
```

Every method trades **memory** against **computation** (recompute, (de)quantize, scoring) or **quality** (lost or blurred information). The thesis measures these three axes for many methods, on the same models and the same inputs.

## 2. Folder structure

```
thesis_lab/
├── configs/lab.yaml          all settings: model, policies, sweeps per experiment
├── kvlab/                    the library (import kvlab)
│   ├── memory_model.py       analytic formulas: KV size, FLOPs, roofline, max context
│   ├── measure.py            correct GPU timing (synchronize) and peak memory
│   ├── compat.py             read/write the HF cache across transformers versions
│   ├── scoring.py            custom attention "kvlab_sdpa": fused SDPA + cheap attention scores
│   ├── models.py             load model/tokenizer, switch attention backend
│   ├── runner.py             KVRunner: prefill/decode loop applying a policy each step
│   ├── data.py               Wikipedia text (en/hy), fixed-length prompts, passkey prompts
│   ├── quality.py            perplexity, bits-per-byte, passkey accuracy
│   ├── io.py                 config + CSV/JSON results
│   └── policies/
│       ├── base.py           KVPolicy interface + helpers
│       ├── budgets.py        per-layer budget schedules (uniform/pyramid/adaptive)
│       ├── eviction.py       SlidingWindow, StreamingLLM, KeyNorm, H2O, TOVA, SnapKV
│       └── quant.py          KIVI (simulated), Tiered mixed precision
├── experiments/              one script per research question (e01 … e07) + plots.py
├── tests/                    pytest suite (tiny random Llama, no download)
└── results/                  CSV/JSON outputs, results/figures/*.png
```

## 3. Getting started

All commands are run from `thesis_lab/`.

```bash
cd thesis_lab
pip install -r requirements.txt        # after installing CUDA PyTorch

# 1. Verify the code (CPU is fine, about a minute, no downloads)
python -m pytest -v

# 2. Analytic study (no GPU needed)
python -m experiments.e01_analytic_memory

# 3. Smoke-test every GPU experiment with tiny sweeps (a few minutes)
python -m experiments.e02_baseline_scaling --quick
python -m experiments.e04_quality --quick
# ... same for e03, e05, e06, e07

# 4. Full runs (can take hours), then the figures
python -m experiments.e02_baseline_scaling
python -m experiments.plots
```

Use `--model meta-llama/Llama-3.1-8B` on any experiment to repeat it with the larger model.

## 4. Methods catalog

### Generic (published) methods

| Method | File | What is traded | Reference |
|---|---|---|---|
| Full cache | `policies/base.py` | baseline, O(T) memory | — |
| No cache (recompute) | `runner.generate_without_cache` | 0 KV memory ↔ O(T) compute per token | — |
| Sliding window | `policies/eviction.py` | memory ↔ loses the beginning of the text | — |
| StreamingLLM | `policies/eviction.py` | keeps "attention sinks" + recent tokens | Xiao et al., ICLR 2024 |
| H2O | `policies/eviction.py` | recent + accumulated heavy hitters | Zhang et al., NeurIPS 2023 |
| TOVA | `policies/eviction.py` | current-query attention only | Oren et al., 2024 |
| SnapKV | `policies/eviction.py` | one-shot prompt compression with an observation window + pooling | Li et al., 2024 |
| Key-norm | `policies/eviction.py` | score-free: low-norm keys attract attention | Devoto et al., EMNLP 2024 |
| PyramidKV schedule | `policies/budgets.py` | more budget in lower layers | Cai et al., 2024 |
| KIVI (simulated) | `policies/quant.py` | 2/4-bit K per-channel, V per-token | Liu et al., ICML 2024 |
| Chunked prefill | `KVRunner(prefill_chunk=…)` | activation memory ↔ prefill speed | — |
| GQA/MQA analysis | `memory_model.py`, e01 | architecture-level KV heads | Ainslie et al. 2023; Shazeer 2019 |
| FlashAttention/SDPA vs eager | e02 | O(T²) → O(T) memory, same FLOPs | Dao et al., 2022 |
| HF static/offloaded/quantized caches | e07 | real memory ↔ PCIe/dequant latency | transformers docs |

### Lab variants (non-generic; candidate thesis contributions)

| Variant | Config name | Hypothesis | Why it might beat the baseline |
|---|---|---|---|
| **Age-normalized H2O** | `h2o_mean` | Dividing accumulated attention by how many queries could see a token removes H2O's bias towards old tokens. | Plain H2O favours early tokens simply because they have been around longer. |
| **Entropy-adaptive layer budgets** | `h2o_adaptive` | Each layer's budget should scale with its *effective number of attended tokens*, exp(entropy). | Layers with peaked attention waste budget under a uniform split; related to CAKE (ICLR 2025), but simpler and parameter-free. |
| **Tiered precision ("demote, don't evict")** | `tiered4`, `tiered2` | At equal memory, keeping unimportant tokens at 2–4 bits beats deleting them, especially for retrieval (e05). | Eviction is irreversible; importance changes over time. Related to MiKV (2024). |
| **Bounded-memory prefill** | e03 part 2 | Evicting between prefill chunks makes peak memory independent of prompt length. | The standard approach compresses only *after* a full prefill, which OOMs first. |
| **Cross-lingual compression cost** | e04 with `languages: [en, hy]` | KV compression hurts Armenian more at the same *token* budget, because each token covers less text. | Measured with bits-per-byte and "KV bytes per byte of text", both tokenizer-independent. Few papers study non-English KV compression. |
| **`logits_to_keep=1` in prefill** | runner + e02 | Computing logits for all T positions costs T × 128k × 4 B (≈ 15.7 GiB at 32k tokens). | A free memory saving that is easy to overlook. |

## 5. Experiments and research questions

| Script | Question | Main output |
|---|---|---|
| `e01_analytic_memory` | How big are the KV cache, attention, and logits vs T? Max context per GPU? Is decoding memory-bound? | analytic tables, roofline |
| `e02_baseline_scaling` | Do measured TTFT, decode latency, and memory follow the analytic model? Eager vs SDPA? | `e02_scaling.csv`, `e02_backends.csv` |
| `e03_chunked_prefill` | Chunk size vs peak memory vs time; bounded-memory prefill up to 128k | `e03_*.csv` |
| `e04_quality` | Quality vs **memory** for every policy, English vs Armenian | `e04_quality.csv` |
| `e05_passkey` | Can the compressed cache still retrieve one specific old fact? | accuracy by depth × length |
| `e06_efficiency` | Does compression speed up or slow down decoding? The no-cache extreme | `e06_*.csv` |
| `e07_hf_caches` | Real memory and latency of static, offloaded, and quantized caches | `e07_hf_caches.csv` |

## 6. Methodology notes (for the thesis "Method" chapter)

1. **Timing:** CUDA is asynchronous. `measure.Timer` synchronizes before and after. A warm-up run precedes each sweep.
2. **Memory:** "peak" = `torch.cuda.max_memory_allocated` (live tensors). It is reset separately for prefill and decode. nvidia-smi shows more, because of the allocator cache.
3. **Held vs logical memory:** `cache_held_bytes` is what is actually on the GPU, counted per unique storage. `cache_logical_bytes` is what a real implementation of the method would need. They differ only for *simulated* quantization (`kivi`, `tiered`), whose quality results are exact but whose storage is still fp16.
4. **Compare at equal memory, not equal budget.** 1,024 fp16 tokens ≈ 4,096 int4 tokens. The e04 plots use logical memory on the x-axis.
5. **Perplexity evaluation:** the text is fed in chunks (`eval_chunk`) and the policy acts between chunks. `chunk=1` is exact streaming but slow. Larger chunks slightly underestimate the damage. Report the value you used.
6. **Cross-lingual metric:** use **bits per byte** (BPB), not perplexity, when comparing languages. Perplexity is per token, and the tokenizer splits Armenian into far more tokens (see `docs/tokenization.md`). The English and Armenian Wikipedia texts are different articles, so they are comparable in distribution, not sentence-by-sentence.
7. **Positions after eviction:** new tokens keep their absolute RoPE position, and cached keys keep theirs. We never re-rotate (unlike StreamingLLM's extrapolation setting), which is valid within Llama-3's 128k trained range. `tests/test_runner.py::test_eviction_equals_masked_attention` proves the implementation equals full attention restricted to the kept tokens.
8. **Scores cost:** score-based policies recompute softmax(qKᵀ) for only the last `score_window` queries: O(w·T) instead of O(T²). For exact H2O accounting, set `score_window ≥ prefill_chunk`.
9. **Batch size 1**, no padding, greedy decoding everywhere. This keeps results deterministic and comparable. Batched serving is future work.
10. **OOM is a result:** sweeps record `"OOM"` instead of crashing. The length at which a method runs out of memory is itself a data point.

## 7. Integrating into the main repository

- `kvlab/` has no dependency on `src/`. You can move the folder as is, or add `thesis_lab` to `PYTHONPATH`.
- The foundation scripts (`src/kv_cache_inspect.py`, etc.) and `kvlab.compat` read the cache the same way. Once you're comfortable, `src/` scripts can import `kvlab.compat.cache_layers` instead of their own `iter_kv`.
- Results go to `thesis_lab/results/`, so they never mix with the Task 1–7 results.

## 8. Extending

- **New policy:** subclass `EvictionPolicy` (implement `_evict`) or `KVPolicy` (implement `compress`), register it in `policies/__init__.py`, and add a named entry in `configs/lab.yaml`. Every experiment can use it immediately. Add a test on hand-made tensors in `tests/test_policies.py`.
- **New model:** any Llama-architecture model works via `--model`. Others (Mistral, Qwen2) should work as long as they use the standard `AttentionInterface`; run the tests with their config.

## 9. Known limitations and next steps

- Quantization is simulated. Real packed storage needs a custom `Cache` layer and a dequantizing kernel; for real numbers, use e07.
- `Tiered` quantizes keys per-token. Combining it with KIVI-style per-channel keys is a natural improvement.
- Eviction happens every step (one gather per step). *Amortized eviction* (evict only when the cache exceeds budget + slack) would cut overhead.
- Per-head adaptive budgets (Ada-KV, 2024) are not implemented. The `budgets.py` interface is the place to add them.
- Language-aware budgets (budget in bytes of text instead of tokens) follow naturally from the e04 cross-lingual results.
- A PagedAttention/vLLM comparison (Kwon et al., SOSP 2023) would add a serving-system perspective.

## 10. References

- Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*
- Shazeer 2019, *Fast Transformer Decoding: One Write-Head is All You Need* (MQA)
- Dao et al. 2022, *FlashAttention*
- Zhang et al. 2023, *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs* (NeurIPS)
- Xiao et al. 2024, *Efficient Streaming Language Models with Attention Sinks* (ICLR)
- Li et al. 2024, *SnapKV: LLM Knows What You are Looking for Before Generation*
- Oren et al. 2024, *Transformers are Multi-State RNNs* (TOVA)
- Devoto et al. 2024, *A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression* (EMNLP)
- Cai et al. 2024, *PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling*
- Liu et al. 2024, *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache* (ICML)
- Yang et al. 2024, *No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization* (MiKV)
- Qin et al. 2025, *CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences* (ICLR)
- Feng et al. 2024, *Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation*
- Mohtashami & Jaggi 2023, *Landmark Attention* (passkey retrieval test)
- Kwon et al. 2023, *Efficient Memory Management for LLM Serving with PagedAttention* (SOSP)
