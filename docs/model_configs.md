# Model Configurations: Llama-3.2-1B vs Llama-3.1-8B

**Script:** [src/model_configs.py](../src/model_configs.py) &nbsp;|&nbsp; **Raw output:** `results/model_configs.json`

The script downloads only `config.json` (a few KB, no weights). `print(config)` gives exactly what `model.config` shows after loading the model.

## 1. Comparison table

> The values below are from the published configs. Check them against the output of `python src/model_configs.py`.

| Field | Llama-3.2-1B | Llama-3.1-8B | Meaning |
|---|---|---|---|
| Layers (`num_hidden_layers`) | 16 | 32 | Number of stacked transformer blocks. Each block has one attention sub-layer and one MLP. |
| Hidden size (`hidden_size`) | 2048 | 4096 | Length of the vector that represents each token between layers (the "residual stream" width). |
| Attention heads (`num_attention_heads`) | 32 | 32 | Number of **query** heads. |
| Key-value heads (`num_key_value_heads`) | 8 | 8 | Number of distinct **key/value** heads. |
| Head dimension (`head_dim`) | 64 | 128 | Size of each head's Q/K/V vectors. `hidden_size / num_attention_heads`. |
| Vocabulary size (`vocab_size`) | 128,256 | 128,256 | Number of token IDs (same tokenizer for both models). |
| MLP size (`intermediate_size`) | 8192 | 14336 | Width of the feed-forward layer. |
| Max positions (`max_position_embeddings`) | 131,072 | 131,072 | 128k-token context window (RoPE positions). |
| GQA group size (heads / kv_heads) | 4 | 4 | Query heads that share one K/V head. |
| KV cache / token (fp16) | 32 KiB | 128 KiB | `2 × layers × kv_heads × head_dim × 2 bytes` |

Derivation of the last row:

$$
\text{KV bytes/token} = \underbrace{2}_{K,V} \times L \times H_{kv} \times d_{head} \times \underbrace{2}_{\text{fp16}}
$$

- 1B: $2 \times 16 \times 8 \times 64 \times 2 = 32{,}768$ B = 32 KiB
- 8B: $2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$ B = 128 KiB

For a full 128k-token context (131,072 tokens), one sequence needs **4 GiB** (1B) or **16 GiB** (8B) of KV cache. For the 8B model that is as much as the fp16 weights themselves (≈ 16 GB).

## 2. Why are there fewer key-value heads than attention heads?

### The mechanism: Grouped-Query Attention (GQA)

In standard **Multi-Head Attention (MHA)**, each of the $H$ heads has its own query, key, and value projection, so $H_q = H_{kv} = H$.

In **Grouped-Query Attention** (Ainslie et al., 2023), the $H_q$ query heads are split into $H_{kv}$ groups. All query heads in a group **share one key head and one value head**. For both Llama models: $H_q = 32$ and $H_{kv} = 8$, so every K/V head serves **4** query heads.

```
MHA  (32 Q, 32 KV):  Q0-K0/V0   Q1-K1/V1   Q2-K2/V2  ...  Q31-K31/V31
GQA  (32 Q,  8 KV):  Q0 Q1 Q2 Q3 -> K0/V0    Q4 Q5 Q6 Q7 -> K1/V1   ...
MQA  (32 Q,  1 KV):  all 32 query heads -> K0/V0
```

GQA sits between MHA ($H_{kv}=H_q$) and **Multi-Query Attention, MQA** ($H_{kv}=1$, Shazeer 2019).

You can see this in the weight shapes. For the 1B model, `q_proj` maps 2048 → 32 × 64 = 2048, but `k_proj` and `v_proj` map 2048 → 8 × 64 = **512**.

### The problem it solves: KV-cache memory and memory bandwidth

During autoregressive generation, the model stores K and V for every past token in every layer (the **KV cache**), so it does not recompute them. The cache size is proportional to $H_{kv}$:

| Llama-3.1-8B, 128k tokens, fp16 | KV heads | KV cache |
|---|---|---|
| If it used MHA | 32 | 64 GiB |
| Actual (GQA) | 8 | **16 GiB** |

So GQA:

1. **Reduces KV-cache memory by** $H_q / H_{kv}$ = 4×. This allows longer contexts or bigger batches on the same GPU.
2. **Speeds up decoding.** Generating each new token means reading the whole KV cache from GPU memory. Decoding is limited by memory bandwidth, not compute, so reading 4× less data makes each step faster.
3. **Keeps quality close to MHA.** Each query head still has its own queries, so heads can attend differently. MQA saves more memory, but loses more quality and is less stable to train.

The trade-off is that K/V heads are shared, so the model has less capacity to represent keys and values. This is a **memory–computation (and memory–quality) trade-off built into the architecture**, which fits directly into the thesis topic.
