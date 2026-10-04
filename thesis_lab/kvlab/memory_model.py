"""
Analytic cost model for decoder-only transformer inference (Llama family).

Why an analytic model?
----------------------
Measurements tell you WHAT happens on one GPU; formulas tell you WHY and let
you extrapolate (other models, 128k context, other GPUs). A good thesis shows
both and checks that they agree (see experiments/e02 vs e01).

Notation
--------
    L    layers                    d     hidden size
    Hq   query heads               Hkv   key/value heads
    Dh   head dim                  V     vocabulary size
    I    MLP intermediate size     T     tokens in context
    B    batch size                b     bytes per element (fp16 = 2)

Key formulas
------------
    KV cache        = 2 * L * Hkv * Dh * T * B * b                    O(T)
    Eager attention = B * Hq * T^2 * b   (score matrix, ONE layer)       O(T^2)
    Prefill logits  = B * T * V * b      (if logits for all positions)   O(T)
    Prefill FLOPs   = 2 * N * T  +  2 * L * Hq * Dh * T * (T + 1)        linear + quadratic
    Decode FLOPs    = 2 * N      +  4 * L * Hq * Dh * t      (at context t)
    Decode bytes    = weights    +  KV(t)          -> decode is memory-bound
where N = number of parameters used in matmuls (all except the embedding lookup).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GiB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    intermediate: int
    tie_embeddings: bool

    @classmethod
    def from_hf_config(cls, cfg, name: str | None = None) -> "ModelSpec":
        """Build from a transformers config object (AutoConfig.from_pretrained)."""
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        return cls(
            name=name or getattr(cfg, "_name_or_path", "model"),
            layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads,
            kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            vocab=cfg.vocab_size,
            intermediate=cfg.intermediate_size,
            tie_embeddings=bool(getattr(cfg, "tie_word_embeddings", False)),
        )

    def with_kv_heads(self, kv_heads: int) -> "ModelSpec":
        """Same model with a different number of KV heads (MHA / GQA / MQA what-if)."""
        return ModelSpec(**{**asdict(self), "kv_heads": kv_heads, "name": f"{self.name}[Hkv={kv_heads}]"})

    # ---- parameter counts ---------------------------------------------------
    @property
    def attn_params_per_layer(self) -> int:
        q = self.hidden * self.heads * self.head_dim
        kv = 2 * self.hidden * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * self.hidden
        return q + kv + o

    @property
    def mlp_params_per_layer(self) -> int:
        return 3 * self.hidden * self.intermediate  # gate, up, down (SwiGLU)

    @property
    def matmul_params(self) -> int:
        """Parameters touched by matrix multiplies per token (includes LM head)."""
        return self.layers * (self.attn_params_per_layer + self.mlp_params_per_layer) + self.vocab * self.hidden

    @property
    def total_params(self) -> int:
        norms = self.layers * 2 * self.hidden + self.hidden
        embed = self.vocab * self.hidden
        lm_head = 0 if self.tie_embeddings else self.vocab * self.hidden
        return self.layers * (self.attn_params_per_layer + self.mlp_params_per_layer) + norms + embed + lm_head


# Published values (check against experiments/e01 which reads the real configs).
LLAMA_32_1B = ModelSpec("Llama-3.2-1B", 16, 2048, 32, 8, 64, 128256, 8192, True)
LLAMA_31_8B = ModelSpec("Llama-3.1-8B", 32, 4096, 32, 8, 128, 128256, 14336, False)
PRESETS = {m.name: m for m in (LLAMA_32_1B, LLAMA_31_8B)}


@dataclass(frozen=True)
class GPUSpec:
    """Datasheet peak numbers (dense fp16 tensor-core FLOP/s, HBM bytes/s).
    Real kernels reach ~50-80% of these; treat roofline results as lower bounds on time."""
    name: str
    memory_gib: float
    fp16_tflops: float
    bandwidth_gbs: float


GPUS = {
    g.name: g
    for g in (
        GPUSpec("T4-16GB", 16, 65, 320),
        GPUSpec("L4-24GB", 24, 121, 300),
        GPUSpec("RTX4090-24GB", 24, 165, 1008),
        GPUSpec("A100-40GB", 40, 312, 1555),
        GPUSpec("A100-80GB", 80, 312, 2039),
        GPUSpec("H100-SXM-80GB", 80, 989, 3350),
    )
}


# ---- memory ----------------------------------------------------------------
def weight_bytes(m: ModelSpec, weight_bits: float = 16) -> float:
    return m.total_params * weight_bits / 8


def kv_bits_per_element(kv_bits: float = 16, group_size: int | None = None) -> float:
    """Effective bits per cached number.

    Integer quantization stores, per group of `group_size` numbers, one fp16
    scale and one fp16 zero-point -> 32 extra bits per group. With 4-bit and
    group 32 the true cost is 4 + 1 = 5 bits, not 4. Papers often forget this.
    """
    if kv_bits >= 16 or not group_size:
        return kv_bits
    return kv_bits + 32 / group_size


def kv_bytes_per_token(m: ModelSpec, kv_bits: float = 16, group_size: int | None = None) -> float:
    return 2 * m.layers * m.kv_heads * m.head_dim * kv_bits_per_element(kv_bits, group_size) / 8


def kv_cache_bytes(m: ModelSpec, tokens: int, batch: int = 1, kv_bits: float = 16,
                   group_size: int | None = None, budget: int | None = None) -> float:
    """KV cache size. `budget` caps stored tokens (eviction / sliding window)."""
    stored = tokens if budget is None else min(tokens, budget)
    return kv_bytes_per_token(m, kv_bits, group_size) * stored * batch


def eager_attention_bytes(m: ModelSpec, tokens: int, batch: int = 1, bytes_per_el: int = 2) -> float:
    """Score matrix of ONE layer in eager attention (B, Hq, T, T). Fused kernels avoid it."""
    return batch * m.heads * tokens * tokens * bytes_per_el


def prefill_logits_bytes(m: ModelSpec, tokens: int, batch: int = 1, bytes_per_el: int = 4) -> float:
    """Logits for every prompt position. Default 4 bytes because many code paths upcast
    logits to fp32. Passing logits_to_keep=1 removes this cost when only the next token
    is needed - a free, often overlooked memory saving at long context."""
    return batch * tokens * m.vocab * bytes_per_el


def max_context(m: ModelSpec, gpu_gib: float, batch: int = 1, weight_bits: float = 16,
                kv_bits: float = 16, group_size: int | None = None, reserve_gib: float = 1.5) -> int:
    """Largest T whose KV cache fits next to the weights (reserve = activations, CUDA context)."""
    free = gpu_gib * GiB - weight_bytes(m, weight_bits) - reserve_gib * GiB
    if free <= 0:
        return 0
    return int(free // (kv_bytes_per_token(m, kv_bits, group_size) * batch))


# ---- compute ----------------------------------------------------------------
def prefill_flops(m: ModelSpec, tokens: int, batch: int = 1) -> dict:
    linear = 2 * m.matmul_params * tokens
    # causal attention: query i looks at i keys -> sum_i i = T(T+1)/2; QK^T and AV each 2*Hq*Dh FLOPs/pair
    attention = 2 * m.layers * m.heads * m.head_dim * tokens * (tokens + 1)
    return {"linear": linear * batch, "attention": attention * batch, "total": (linear + attention) * batch}


def decode_step_flops(m: ModelSpec, context: int, batch: int = 1) -> dict:
    linear = 2 * m.matmul_params
    attention = 4 * m.layers * m.heads * m.head_dim * context
    return {"linear": linear * batch, "attention": attention * batch, "total": (linear + attention) * batch}


def decode_step_roofline(m: ModelSpec, gpu: GPUSpec, context: int, batch: int = 1, weight_bits: float = 16,
                         kv_bits: float = 16, group_size: int | None = None) -> dict:
    """Lower-bound time of ONE decode step: max(compute time, memory-traffic time).

    Every step must read all weights once (shared by the batch) and the whole
    KV cache of every sequence. Arithmetic intensity (FLOPs per byte) far below
    the GPU's ratio (e.g. A100: 312e12 / 1.555e12 ~ 200) means memory-bound:
    shrinking bytes (GQA, KV quantization, eviction) directly speeds decoding.
    """
    flops = decode_step_flops(m, context, batch)["total"]
    bytes_moved = weight_bytes(m, weight_bits) + kv_cache_bytes(m, context, batch, kv_bits, group_size)
    t_compute = flops / (gpu.fp16_tflops * 1e12)
    t_memory = bytes_moved / (gpu.bandwidth_gbs * 1e9)
    return {
        "flops": flops,
        "bytes": bytes_moved,
        "intensity_flops_per_byte": flops / bytes_moved,
        "gpu_balance_flops_per_byte": gpu.fp16_tflops * 1e12 / (gpu.bandwidth_gbs * 1e9),
        "t_compute_ms": t_compute * 1e3,
        "t_memory_ms": t_memory * 1e3,
        "t_lower_bound_ms": max(t_compute, t_memory) * 1e3,
        "bound": "memory" if t_memory > t_compute else "compute",
    }


def recompute_vs_cache(m: ModelSpec, prompt: int, new_tokens: int) -> dict:
    """The extreme trade-off: generating WITHOUT a KV cache.

    With cache   : prefill once, then each step processes 1 token.
    Without cache: each step re-runs the full forward over all t tokens so far.
    Memory saved = the whole KV cache; compute paid = roughly new_tokens full prefills.
    """
    with_cache = prefill_flops(m, prompt)["total"] + sum(
        decode_step_flops(m, prompt + i)["total"] for i in range(new_tokens))
    without = sum(prefill_flops(m, prompt + i + 1)["total"] for i in range(new_tokens))
    return {
        "flops_with_cache": with_cache,
        "flops_without_cache": without,
        "compute_ratio": without / with_cache,
        "kv_bytes_saved_fp16": kv_cache_bytes(m, prompt + new_tokens),
    }
