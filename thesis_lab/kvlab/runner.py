"""
KVRunner: prefill + greedy decoding with a KV-cache policy applied after every forward pass.

Why not model.generate()?
-------------------------
generate() hides the cache. To study the cache we need to modify it between
forward passes and measure each phase separately, so we write the loop ourselves
(~50 lines). This is the same loop generate() runs internally for greedy decoding.

Two kinds of positions (the key to making eviction correct)
-----------------------------------------------------------
After eviction the cache holds FEWER entries than tokens seen. So:
    position_ids   = absolute position in the text  -> used for RoPE
    cache_position = index where new entries land in the (compressed) cache
                     -> used by transformers for mask bookkeeping
Example: 10,000 tokens seen, cache compressed to 1,024 entries. The next token
has position_ids = 10000 (its true place in the text, so relative distances to
the kept keys stay correct) but cache_position = 1024.

Phases and what is measured
---------------------------
    prefill : whole prompt (optionally in chunks) -> TTFT (time to first token)
    decode  : one token per forward               -> per-token latency (TPOT)
Peak GPU memory is recorded separately for each phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from . import compat, measure, scoring
from .policies import FullCache, KVPolicy


@dataclass
class GenerationResult:
    prompt_tokens: int
    new_token_ids: list[int]
    ttft_s: float
    decode_step_s: list[float] = field(default_factory=list)
    peak_prefill_bytes: int | None = None
    peak_decode_bytes: int | None = None
    cache_tokens_per_layer: list[int] = field(default_factory=list)
    cache_held_bytes: int = 0
    cache_logical_bytes: float = 0.0

    def summary(self) -> dict:
        steps = torch.tensor(self.decode_step_s) if self.decode_step_s else None
        return {
            "prompt_tokens": self.prompt_tokens,
            "new_tokens": len(self.new_token_ids),
            "ttft_s": self.ttft_s,
            "decode_ms_mean": steps.mean().item() * 1e3 if steps is not None else None,
            "decode_ms_median": steps.median().item() * 1e3 if steps is not None else None,
            "decode_tokens_per_s": 1.0 / steps.mean().item() if steps is not None else None,
            "peak_prefill_gib": measure.to_gib(self.peak_prefill_bytes),
            "peak_decode_gib": measure.to_gib(self.peak_decode_bytes),
            "cache_tokens_mean": sum(self.cache_tokens_per_layer) / max(1, len(self.cache_tokens_per_layer)),
            "cache_held_mib": measure.to_mib(self.cache_held_bytes),
            "cache_logical_mib": measure.to_mib(self.cache_logical_bytes),
        }

    def to_dict(self) -> dict:
        return asdict(self)


class KVRunner:
    """
    runner = KVRunner(model, policy, score_window=32, prefill_chunk=2048)
    result = runner.generate(input_ids, max_new_tokens=64)
    nll    = runner.nll(input_ids, chunk=32)       # for perplexity

    score_window  : how many of the newest queries per forward contribute
                    attention statistics (SnapKV observation window). Only used
                    by policies with needs_scores=True.
    prefill_chunk : process the prompt in chunks of this size (None = one pass).
                    Smaller chunks -> lower activation memory, and eviction can
                    run between chunks -> bounded memory for any prompt length.
    """

    def __init__(self, model, policy: KVPolicy | None = None, score_window: int = 32,
                 prefill_chunk: int | None = None):
        self.model = model
        self.policy = policy or FullCache()
        self.score_window = score_window
        self.prefill_chunk = prefill_chunk
        self.device = next(model.parameters()).device

        attn = getattr(model.config, "_attn_implementation", None)
        if (self.policy.needs_scores or not self.policy.uniform_lengths) and attn != scoring.ATTN_NAME:
            raise ValueError(
                f"Policy {self.policy.name!r} needs attention scores or per-layer cache lengths; "
                f"load the model with attn='{scoring.ATTN_NAME}' (current: {attn!r}).")
        self.policy.setup(model.config.num_hidden_layers)
        self.reset()

    def reset(self) -> None:
        """Start a new sequence: empty cache, position 0, fresh policy state."""
        self.cache = compat.new_cache()
        self.pos = 0
        self.policy.reset()
        scoring.STORE.clear()

    # ---- core ---------------------------------------------------------------
    @torch.no_grad()
    def _forward(self, ids: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        n = ids.shape[1]
        layers = compat.cache_layers(self.cache) if self.pos else []
        start = layers[0][0].shape[2] if layers else 0
        scoring.STORE.configure(self.policy.needs_scores, self.score_window)
        scoring.STORE.clear()

        out = self.model(
            input_ids=ids,
            past_key_values=self.cache,
            use_cache=True,
            position_ids=torch.arange(self.pos, self.pos + n, device=self.device).unsqueeze(0),
            cache_position=torch.arange(start, start + n, device=self.device),
            logits_to_keep=logits_to_keep,  # 1 -> skip the (T x vocab) logits matrix during prefill
        )
        self.cache = out.past_key_values
        self.pos += n
        self._apply_policy(n)
        return out.logits

    def _apply_policy(self, new_tokens: int) -> None:
        scores = scoring.STORE.scores
        self.policy.begin_step(new_tokens, min(self.score_window, new_tokens), scores)
        for i, (k, v) in enumerate(compat.cache_layers(self.cache)):
            k2, v2 = self.policy.compress(i, k, v, scores.get(i), new_tokens)
            if k2 is not k or v2 is not v:
                compat.set_layer(self.cache, i, k2, v2)
        scoring.STORE.clear()

    # ---- public API -----------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Process the prompt; returns next-token logits (B, V)."""
        T = input_ids.shape[1]
        chunk = self.prefill_chunk or T
        for s in range(0, T, chunk):
            logits = self._forward(input_ids[:, s:s + chunk], logits_to_keep=1)
        return logits[:, -1, :]

    @torch.no_grad()
    def step(self, token_ids: torch.Tensor) -> torch.Tensor:
        """One decode step for token_ids (B, 1); returns next-token logits (B, V)."""
        return self._forward(token_ids, logits_to_keep=1)[:, -1, :]

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id=None) -> GenerationResult:
        """Greedy generation. The first new token comes from the prefill logits (TTFT);
        each further token costs one decode step."""
        eos = set() if eos_token_id is None else set(eos_token_id if isinstance(eos_token_id, (list, tuple))
                                                     else [eos_token_id])
        self.reset()
        measure.reset_peak(self.device)
        with measure.Timer(self.device) as t:
            nxt = self.prefill(input_ids).argmax(dim=-1, keepdim=True)
        result = GenerationResult(prompt_tokens=input_ids.shape[1], new_token_ids=[nxt.item()], ttft_s=t.seconds,
                                  peak_prefill_bytes=measure.peak_bytes(self.device))

        measure.reset_peak(self.device)
        while len(result.new_token_ids) < max_new_tokens and result.new_token_ids[-1] not in eos:
            with measure.Timer(self.device) as t:
                nxt = self.step(nxt).argmax(dim=-1, keepdim=True)
            result.decode_step_s.append(t.seconds)
            result.new_token_ids.append(nxt.item())
        result.peak_decode_bytes = measure.peak_bytes(self.device)

        layers = compat.cache_layers(self.cache)
        result.cache_tokens_per_layer = [k.shape[2] for k, _ in layers]
        result.cache_held_bytes = compat.cache_bytes(self.cache)
        result.cache_logical_bytes = self.policy.logical_bytes(layers)
        return result

    @torch.no_grad()
    def nll(self, input_ids: torch.Tensor, chunk: int = 32) -> torch.Tensor:
        """Negative log-likelihood of every token after the first, (T-1,) float32.

        The text is fed in chunks; the policy compresses the cache between chunks.
        chunk=1 is exact streaming evaluation (the policy acts before EVERY
        prediction) but slow; within a chunk tokens still see each other fully,
        so larger chunks slightly UNDER-estimate the damage of eviction.
        Report the chunk size you used.
        """
        self.reset()
        T = input_ids.shape[1]
        out = []
        for s in range(0, T - 1, chunk):
            e = min(s + chunk, T - 1)
            logits = self._forward(input_ids[:, s:e], logits_to_keep=0)  # 0 = keep all positions
            target = input_ids[:, s + 1:e + 1]
            logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            out.append(-logp)
        return torch.cat(out, dim=1)[0]

    def cache_stats(self) -> dict:
        layers = compat.cache_layers(self.cache)
        return {
            "tokens_seen": self.pos,
            "cache_tokens_per_layer": [k.shape[2] for k, _ in layers],
            "held_bytes": compat.cache_bytes(self.cache),
            "logical_bytes": self.policy.logical_bytes(layers),
        }


@torch.no_grad()
def generate_without_cache(model, input_ids: torch.Tensor, max_new_tokens: int) -> dict:
    """The compute extreme of the trade-off: NO KV cache. Every step re-runs the
    whole sequence, so step t costs a full prefill of t tokens (O(t^2) attention),
    while KV memory is zero."""
    device = input_ids.device
    ids, times = input_ids, []
    measure.reset_peak(device)
    for _ in range(max_new_tokens):
        with measure.Timer(device) as t:
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            ids = torch.cat([ids, logits.argmax(dim=-1, keepdim=True)], dim=1)
        times.append(t.seconds)
    return {"step_s": times, "peak_bytes": measure.peak_bytes(device),
            "new_token_ids": ids[0, input_ids.shape[1]:].tolist()}
