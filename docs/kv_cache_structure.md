# KV Cache Structure

**Script:** [src/kv_cache_inspect.py](../src/kv_cache_inspect.py) &nbsp;|&nbsp; **Raw output:** `results/kv_cache.json`

Model: `meta-llama/Llama-3.2-1B`, fp16. Prompt: "The capital of France is" → $T = 6$ tokens (including BOS).

## 1. Why a KV cache exists

A causal LM generates one token at a time. At step $t$, the new token's query must attend to the **keys and values of all previous tokens**. Those K/V vectors do not change once computed, because of the causal mask: past tokens never see future ones. So instead of recomputing them at every step, the model **stores them**:

- **Prefill:** process the whole prompt in parallel and store K and V for all $T$ tokens in every layer.
- **Decode:** for each new token, compute Q/K/V for **only that token**, append its K/V to the cache, and attend over the full cache.

This changes the compute per generated token from re-processing all $T$ tokens to processing 1 token. In exchange, it costs **memory that grows linearly with $T$**. This is the central memory–computation trade-off of the thesis.

## 2. What `past_key_values` contains

In recent `transformers` versions, `out.past_key_values` is a `DynamicCache` object. The script reads it through `iter_kv()`, which handles all versions.

- **Number of entries: 16** (one per layer, $L = 16$). Each entry holds **one key tensor and one value tensor**, so 32 tensors in total.
- **Shape of every tensor:** `(1, 8, 6, 64)`.

> TODO: paste the printed shapes from `python src/kv_cache_inspect.py` here.

After one decode step, the shape becomes `(1, 8, 7, 64)`. Only the sequence axis grows.

## 3. Mapping each dimension to the model config (Task 4)

| Dim | Size | Config field | Explanation |
|---|---|---|---|
| number of entries | 16 | `num_hidden_layers` | every layer has its own attention, so its own K/V |
| 0 – batch | 1 | — | one prompt |
| 1 – heads | **8** | `num_key_value_heads` | **not** 32: with GQA only the 8 shared K/V heads are stored. The 4× repeat to 32 query heads happens on the fly inside attention. |
| 2 – sequence | 6 → 7 → … | — (input length) | one slot per token seen so far; grows by 1 per generated token |
| 3 – head dim | 64 | `head_dim` = `hidden_size / num_attention_heads` = 2048 / 32 | length of each key/value vector |

Keys are stored **after RoPE** is applied, so each cached key already contains its position information. Values have no positional encoding.

Consistency check with the projection weights: `k_proj` outputs $8 \times 64 = 512$ features per token, which is exactly one row of one layer's key cache.

## 4. Memory

$$
\text{KV bytes} = \underbrace{2}_{K,V} \times \underbrace{L}_{16} \times \underbrace{H_{kv}}_{8} \times \underbrace{d_{head}}_{64} \times T \times \underbrace{2}_{\text{fp16}} = 32{,}768 \cdot T \text{ bytes}
$$

The script checks that the measured bytes equal this formula.

| Tokens $T$ | Llama-3.2-1B | Llama-3.1-8B (128 KiB/token) |
|---|---|---|
| 6 (our prompt) | 192 KiB | 768 KiB |
| 1,024 | 32 MiB | 128 MiB |
| 32,768 | 1 GiB | 4 GiB |
| 131,072 (128k) | **4 GiB** | **16 GiB** |

These numbers are **per sequence**. They scale linearly with batch size as well, so serving many long-context requests quickly fills GPU memory.

## 5. Takeaways for the thesis

- The KV cache grows as $O(L \cdot H_{kv} \cdot d_{head} \cdot T \cdot B)$, **linear in context length**. Attention scores without a cache or fused kernel would be $O(T^2)$.
- Every decode step must **read the whole cache** from GPU memory, so long-context decoding is limited by memory bandwidth.
- Ways to trade memory against compute or quality, all of which act on one of the dimensions above:
  - fewer KV heads (GQA/MQA) → dim 1
  - lower precision (KV quantization) → bytes per element
  - dropping or evicting tokens (sliding window, H2O, StreamingLLM) → dim 2
  - sharing across layers → number of entries
  - offloading to CPU, or recomputing instead of storing
