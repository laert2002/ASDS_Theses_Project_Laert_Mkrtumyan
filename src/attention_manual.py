"""
Task 6 - Scaled dot-product attention implemented by hand.

What it does
------------
Implements attention with plain tensor ops, then checks numerically that it
gives the same result as PyTorch's fused
torch.nn.functional.scaled_dot_product_attention (SDPA).

Shapes follow Llama-3.2-1B: 32 query heads, 8 key/value heads (GQA), head_dim 64.

Run (no GPU or model download needed):
    python src/attention_manual.py

The formula
-----------
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_head) + M ) V

    Q : (B, H_q,  T_q, d_head)  queries  - "what am I looking for?"
    K : (B, H_kv, T_k, d_head)  keys     - "what do I contain?"
    V : (B, H_kv, T_k, d_head)  values   - "what do I pass on if selected?"
    M : causal mask, 0 where key_pos <= query_pos, -inf otherwise
"""

import math

import torch
import torch.nn.functional as F


def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """Scaled dot-product attention with optional causal mask and GQA support."""
    B, Hq, Tq, D = q.shape
    Hkv, Tk = k.shape[1], k.shape[2]

    # GQA: each K/V head is shared by Hq // Hkv query heads. repeat_interleave
    # turns [kv0, kv1, ...] into [kv0, kv0, kv0, kv0, kv1, ...] so query head h
    # uses KV head h // group. (Same thing HF's `repeat_kv` does.)
    group = Hq // Hkv
    k = k.repeat_interleave(group, dim=1)  # (B, Hq, Tk, D)
    v = v.repeat_interleave(group, dim=1)

    # 1) Similarity of every query with every key: (B, Hq, Tq, Tk).
    #    Divide by sqrt(D) so the dot products don't grow with D; otherwise
    #    softmax saturates and gradients vanish.
    scores = q @ k.transpose(-2, -1) / math.sqrt(D)

    # 2) Causal mask: query i may only see keys j <= i.
    if causal:
        allowed = torch.ones(Tq, Tk, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))

    # 3) Softmax over the key axis -> each row is a probability distribution.
    weights = torch.softmax(scores, dim=-1)

    # 4) Weighted average of values: (B, Hq, Tq, Tk) @ (B, Hq, Tk, D) -> (B, Hq, Tq, D).
    return weights @ v


def check(causal: bool, B=1, Hq=32, Hkv=8, T=128, D=64, dtype=torch.float32, device="cpu") -> bool:
    q = torch.randn(B, Hq, T, D, dtype=dtype, device=device)
    k = torch.randn(B, Hkv, T, D, dtype=dtype, device=device)
    v = torch.randn(B, Hkv, T, D, dtype=dtype, device=device)

    ours = manual_attention(q, k, v, causal=causal)

    # Reference: expand K/V to Hq heads ourselves so the check works on every
    # torch version (enable_gqa=True exists only in torch >= 2.5).
    group = Hq // Hkv
    ref = F.scaled_dot_product_attention(
        q, k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1), is_causal=causal
    )

    # Floating-point results differ slightly depending on the order of
    # operations, so we compare with a tolerance instead of exact equality.
    atol = 1e-5 if dtype == torch.float32 else 2e-3
    max_diff = (ours.float() - ref.float()).abs().max().item()
    ok = torch.allclose(ours, ref, atol=atol, rtol=1e-3)
    print(f"causal={causal!s:<5} dtype={str(dtype):<14} device={device:<4} "
          f"shape={tuple(ours.shape)}  max|diff|={max_diff:.2e}  match={ok}")
    return ok


def main() -> None:
    torch.manual_seed(0)
    results = [check(causal=False), check(causal=True)]
    if torch.cuda.is_available():
        # fp16 on GPU is what the real model uses; tolerance must be looser.
        results += [check(causal=True, dtype=torch.float16, device="cuda")]
    print("ALL MATCH" if all(results) else "MISMATCH FOUND")


if __name__ == "__main__":
    main()
