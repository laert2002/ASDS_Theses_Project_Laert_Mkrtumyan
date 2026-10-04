# Tensor Shapes in a Forward Pass

**Script:** [src/forward_shapes.py](../src/forward_shapes.py) &nbsp;|&nbsp; **Raw output:** `results/forward_shapes.json`

Model: `meta-llama/Llama-3.2-1B`, fp16, loaded with `attn_implementation="eager"`. The default SDPA/Flash kernels never build the full attention matrix in memory, so they cannot return it with `output_attentions=True`.

Notation used below (numbers for Llama-3.2-1B):

| Symbol | Meaning | Value |
|---|---|---|
| $B$ | batch size | 1 |
| $T$ | sequence length in tokens (including BOS) | 6 for our prompt* |
| $L$ | number of layers | 16 |
| $d$ | hidden size | 2048 |
| $H$ | attention (query) heads | 32 |
| $V$ | vocabulary size | 128,256 |

\* "The capital of France is" → `<|begin_of_text|>`, `The`, `Ġcapital`, `Ġof`, `ĠFrance`, `Ġis`. Check this against the script output.

## 1. Input

`input_ids`: **(B, T)** = (1, 6). One integer token ID per position.

## 2. Hidden states: `out.hidden_states`

A tuple of **L + 1 = 17** tensors, each of shape **(B, T, d)** = (1, 6, 2048).

| Index | What it is |
|---|---|
| `[0]` | Output of the embedding lookup (input to layer 1). No position information is added here, because Llama applies RoPE inside attention. |
| `[1]` … `[15]` | Output (residual stream) after transformer layers 1 … 15. |
| `[16]` | Output of layer 16 **after the final RMSNorm**. This is what the LM head uses. |

Dimensions:
- **B**: which sequence in the batch.
- **T**: which token position. Every token keeps its own vector through all layers.
- **d**: the features of that token's representation. The shape stays constant through the network, because every layer reads from and writes back into this residual stream.

## 3. Attention matrices: `out.attentions`

A tuple of **L = 16** tensors, one per layer, each of shape **(B, H, T, T)** = (1, 32, 6, 6).

| Dim | Meaning |
|---|---|
| **B** | batch index |
| **H = 32** | attention head. It is 32, not 8 KV heads: with GQA, each K/V head is repeated for its 4 query heads before the softmax, so every **query** head gets its own attention pattern. |
| **T (3rd)** | **query** position $i$, the token that is "looking" |
| **T (4th)** | **key** position $j$, the token being looked at |

Entry `[b, h, i, j]` = how much token $i$ attends to token $j$ in head $h$:

$$
A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_{head}}} + M\right), \qquad M_{ij} = \begin{cases}0 & j \le i \\ -\infty & j > i\end{cases}
$$

The script checks two properties:
- **Each row sums to 1**, because it is a probability distribution over keys.
- **The upper triangle is 0**, because of the causal mask $M$: a token cannot attend to future tokens.

## 4. Logits: `out.logits`

**(B, T, V)** = (1, 6, 128256). For every position, one unnormalised score per vocabulary token for the **next** token. Generation only uses `logits[:, -1, :]`.

## 5. How does the attention matrix grow with input length?

Per layer, the attention matrix has $B \cdot H \cdot T^2$ elements, so it grows **quadratically**, $O(T^2)$. Doubling the input length gives 4× the memory and 4× the $QK^\top$ compute.

For Llama-3.2-1B in fp16 ($H = 32$, 2 bytes, $B = 1$): bytes per layer = $32 \cdot T^2 \cdot 2 = 64\,T^2$.

| T | Shape per layer | Per layer | All 16 layers |
|---|---|---|---|
| 16 | (1, 32, 16, 16) | 16 KiB | 256 KiB |
| 64 | (1, 32, 64, 64) | 256 KiB | 4 MiB |
| 256 | (1, 32, 256, 256) | 4 MiB | 64 MiB |
| 512 | (1, 32, 512, 512) | 16 MiB | 256 MiB |
| 131,072 (128k) | (1, 32, 131072, 131072) | **1 TiB** | 16 TiB |

> TODO: confirm the first four rows with the script output (Part B).

**Consequences for long context:**

- Building the full $T \times T$ matrix is impossible at long context lengths. FlashAttention / SDPA kernels compute attention in tiles and never store the full matrix, which brings memory down to $O(T)$. **Compute stays $O(T^2)$**, which is a memory–computation trade-off: some values are recomputed instead of stored.
- The hidden states grow only **linearly**, $O(T \cdot d)$.
- During generation, a new token has $T_q = 1$, so its attention row is only $(B, H, 1, T)$, which is linear in $T$. But it needs the K and V of **all** previous tokens, and that is what the KV cache stores (see [kv_cache_structure.md](kv_cache_structure.md)).
