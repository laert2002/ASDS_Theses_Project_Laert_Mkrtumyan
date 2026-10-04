"""End-to-end checks of KVRunner + kvlab_sdpa on a tiny RANDOM Llama.

No download and no GPU needed: the model is built from a config with random
weights. These tests catch the bugs that would silently corrupt thesis results:
wrong positions after eviction, wrong causal masks during chunked prefill, and
attention scores that are not probability mass.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from kvlab import compat, scoring  # noqa: E402
from kvlab.policies import FullCache, KVPolicy, SlidingWindow, StreamingLLM, build  # noqa: E402
from kvlab.runner import KVRunner  # noqa: E402

VOCAB, LAYERS, HEADS, KV_HEADS = 101, 3, 4, 2  # head_dim = 64 / 4 = 16, GQA group = 2


@pytest.fixture(scope="module")
def models():
    scoring.register()
    cfg = transformers.LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                                   num_key_value_heads=KV_HEADS, max_position_embeddings=512,
                                   initializer_range=0.2, tie_word_embeddings=False)
    torch.manual_seed(0)
    sdpa = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").eval()
    kvlab = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation=scoring.ATTN_NAME).eval()
    kvlab.load_state_dict(sdpa.state_dict())
    return {"sdpa": sdpa, "kvlab": kvlab}


@pytest.fixture
def ids():
    return torch.randint(0, VOCAB, (1, 48), generator=torch.Generator().manual_seed(1))


def close(a, b):
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@torch.no_grad()
def test_kvlab_attention_matches_sdpa(models, ids):
    scoring.STORE.configure(False, 32)
    close(models["kvlab"](ids).logits, models["sdpa"](ids).logits)


@pytest.mark.parametrize("which", ["sdpa", "kvlab"])
@torch.no_grad()
def test_cached_decoding_matches_full_recompute(models, ids, which):
    """Chunked prefill + step-by-step decoding must equal recomputing from scratch."""
    model = models[which]
    runner = KVRunner(model, FullCache(), prefill_chunk=7)
    close(runner.prefill(ids[:, :30]), model(ids[:, :30], use_cache=False).logits[:, -1])
    for t in range(30, 48):
        close(runner.step(ids[:, t:t + 1]), model(ids[:, :t + 1], use_cache=False).logits[:, -1])


@torch.no_grad()
def test_cache_layout(models, ids):
    runner = KVRunner(models["kvlab"])
    runner.prefill(ids[:, :20])
    layers = compat.cache_layers(runner.cache)
    assert len(layers) == LAYERS
    assert all(k.shape == v.shape == (1, KV_HEADS, 20, 16) for k, v in layers)


def reference_nll(model, ids, allowed):
    """NLL from ONE full forward pass with an explicit (T, T) visibility mask."""
    T = ids.shape[1]
    mask = torch.zeros(1, 1, T, T)
    mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)
    logits = model(ids, attention_mask=mask, use_cache=False).logits
    return -torch.log_softmax(logits[0, :-1], -1).gather(-1, ids[0, 1:, None]).squeeze(-1)


@pytest.mark.parametrize("which", ["sdpa", "kvlab"])
@pytest.mark.parametrize("sink", [0, 4])
@torch.no_grad()
def test_eviction_equals_masked_attention(models, ids, which, sink):
    """Streaming eviction must equal full attention restricted to the kept tokens.

    With budget b and chunk=1, token i sees keys j <= i with j < sink or j >= i - (b - sink).
    Passing this proves positions (RoPE), cache bookkeeping and masking are all correct.
    """
    b = 12
    policy = StreamingLLM(budget=b, sink=sink) if sink else SlidingWindow(budget=b)
    got = KVRunner(models[which], policy).nll(ids, chunk=1)

    i = torch.arange(48)[:, None]
    j = torch.arange(48)[None, :]
    allowed = (j <= i) & ((j < sink) | (j >= i - (b - sink)))
    close(got, reference_nll(models["sdpa"], ids, allowed))


@torch.no_grad()
def test_large_budget_changes_nothing(models, ids):
    full = KVRunner(models["kvlab"]).nll(ids, chunk=8)
    for spec in ({"type": "h2o"}, {"type": "snapkv"}, {"type": "sliding_window"}):
        close(KVRunner(models["kvlab"], build(spec, 1000), score_window=4).nll(ids, chunk=8), full)


class Probe(KVPolicy):
    needs_scores = True

    def reset(self):
        self.mass = []

    def compress(self, layer, key, value, scores, new_tokens):
        assert scores.shape == (1, KV_HEADS, key.shape[2])
        self.mass.append(scores.sum(-1))
        return key, value


@torch.no_grad()
def test_scores_are_attention_mass(models, ids):
    """Each KV head's scores sum to (query heads per KV head) x (observation queries)."""
    probe = Probe()
    runner = KVRunner(models["kvlab"], probe, score_window=4)
    runner.prefill(ids[:, :20])
    group = HEADS // KV_HEADS
    for m in probe.mass:
        close(m, torch.full_like(m, group * 4.0))
    probe.mass.clear()
    runner.step(ids[:, 20:21])
    for m in probe.mass:
        close(m, torch.full_like(m, group * 1.0))


SPECS = {
    "h2o": {"type": "h2o"},
    "h2o_mean": {"type": "h2o", "aggregation": "mean"},
    "tova": {"type": "tova"},
    "keynorm": {"type": "keynorm"},
    "h2o_pyramid": {"type": "h2o", "schedule": "pyramid"},
    "h2o_adaptive": {"type": "h2o", "schedule": "adaptive"},
    "snapkv": {"type": "snapkv", "kernel": 3},
    "kivi": {"type": "kivi", "bits": 4, "group_size": 8, "residual": 8},
    "tiered": {"type": "tiered", "low_bits": 4, "group_size": 8},
}


@pytest.mark.parametrize("name", SPECS)
@torch.no_grad()
def test_policies_generate_and_respect_budgets(models, ids, name):
    policy = build(SPECS[name], 16)
    res = KVRunner(models["kvlab"], policy, score_window=4, prefill_chunk=8).generate(ids[:, :40], 6)
    assert len(res.new_token_ids) == 6
    seen = 40 + 5  # prompt + decode steps (the first new token comes from the prefill logits)
    lengths = res.cache_tokens_per_layer
    if name in ("kivi", "tiered"):
        assert lengths == [seen] * LAYERS
        assert res.cache_logical_bytes < res.cache_held_bytes
    elif name == "snapkv":
        assert lengths == [16 + 5] * LAYERS  # prompt compressed to budget, decode tokens kept
    else:
        assert lengths == [min(seen, policy.budget_of(i)) for i in range(LAYERS)]
    if name == "h2o_pyramid":
        assert lengths[0] > lengths[-1]


@torch.no_grad()
def test_greedy_matches_hf_generate(models, ids):
    model = models["sdpa"]
    eos = model.generation_config.eos_token_id
    hf = model.generate(ids[:, :20], max_new_tokens=10, do_sample=False, pad_token_id=0)[0, 20:].tolist()
    ours = KVRunner(model).generate(ids[:, :20], 10, eos_token_id=eos).new_token_ids
    assert ours == hf


def test_score_policy_requires_kvlab_attention(models):
    with pytest.raises(ValueError):
        KVRunner(models["sdpa"], build({"type": "h2o"}, 16))
