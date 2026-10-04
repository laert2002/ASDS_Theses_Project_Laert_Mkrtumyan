"""Policy logic on hand-made tensors (pure PyTorch, no model needed).

Trick used throughout: fill every key/value vector with its POSITION, so after
compression `k[0, 0, :, 0]` lists exactly which tokens survived.
"""

import pytest
import torch

from kvlab.policies import (H2O, KIVI, TOVA, FullCache, KeyNorm, SlidingWindow, SnapKV, StreamingLLM, Tiered,
                            budgets, build)
from kvlab.policies.quant import quant_per_channel, quant_per_token

B, H, D = 1, 2, 64


def positional_kv(T):
    k = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1).expand(B, H, T, D).clone()
    return k, k.clone()


def random_kv(T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, H, T, D, generator=g), torch.randn(B, H, T, D, generator=g)


def run(policy, k, v, scores=None, new_tokens=None, obs=1, layers=1):
    policy.setup(layers)
    new_tokens = new_tokens or k.shape[2]
    policy.begin_step(new_tokens, obs, {0: scores} if scores is not None else {})
    return policy.compress(0, k, v, scores, new_tokens)


def kept(k):
    return k[0, 0, :, 0].long().tolist()


def peaked_scores(T, hot):
    s = torch.full((B, H, T), 0.01)
    s[..., hot] = 1.0
    return s


def test_full_cache_keeps_everything():
    k, v = positional_kv(20)
    k2, v2 = run(FullCache(), k, v)
    assert k2 is k and v2 is v


def test_under_budget_is_untouched():
    k, v = positional_kv(8)
    k2, _ = run(SlidingWindow(budget=16), k, v)
    assert k2 is k


def test_sliding_window_keeps_last():
    k, v = positional_kv(20)
    k2, v2 = run(SlidingWindow(budget=8), k, v)
    assert kept(k2) == list(range(12, 20)) and kept(v2) == list(range(12, 20))


def test_streaming_llm_keeps_sinks_and_recent():
    k, v = positional_kv(20)
    k2, _ = run(StreamingLLM(budget=10, sink=4), k, v)
    assert kept(k2) == [0, 1, 2, 3] + list(range(14, 20))


def test_h2o_keeps_heavy_hitters_and_recent():
    k, v = positional_kv(20)
    policy = H2O(budget=6, recent=0.5)
    k2, _ = run(policy, k, v, peaked_scores(20, [2, 5, 7]))
    assert kept(k2) == [2, 5, 7, 17, 18, 19]
    assert policy.acc[0].shape[-1] == 6  # statistics follow the kept tokens


def test_tova_uses_only_current_scores():
    k, v = positional_kv(12)
    k2, _ = run(TOVA(budget=3), k, v, peaked_scores(12, [1, 4, 9]))
    assert kept(k2) == [1, 4, 9]


def test_mean_aggregation_counts_visible_queries():
    k, v = positional_kv(10)
    policy = H2O(budget=64, aggregation="mean")
    run(policy, k, v, torch.ones(B, H, 10), obs=4)
    # last 4 tokens are seen by 4, 3, 2, 1 observation queries (causal)
    assert policy.seen[0][0, 0].tolist() == [4] * 6 + [4, 3, 2, 1]


def test_keynorm_keeps_smallest_keys():
    k, v = random_kv(12)
    k[:, :, [3, 8]] *= 0.01
    k2, _ = run(KeyNorm(budget=2), k, v)
    assert torch.equal(k2, k[:, :, [3, 8]])


def test_snapkv_keeps_pooled_prefix_and_window():
    k, v = positional_kv(20)
    k2, _ = run(SnapKV(budget=7, kernel=3), k, v, peaked_scores(20, [5]), new_tokens=20, obs=4)
    assert kept(k2) == [4, 5, 6, 16, 17, 18, 19]  # neighbours of 5 kept via pooling


def test_snapkv_does_not_evict_during_decode():
    k, v = positional_kv(20)
    k2, _ = run(SnapKV(budget=7), k, v, peaked_scores(20, [5]), new_tokens=1)
    assert k2 is k


def test_quantization_error_shrinks_with_bits():
    x = torch.randn(1, 2, 64, 64)
    err = {b: (quant_per_token(x, b, 32) - x).abs().mean().item() for b in (2, 4, 8)}
    assert err[8] < err[4] < err[2]
    assert err[8] / x.abs().mean().item() < 0.01
    assert (quant_per_channel(x, 8, 32) - x).abs().mean().item() / x.abs().mean().item() < 0.01


def test_kivi_quantizes_only_full_blocks_outside_residual():
    k, v = random_kv(200)
    k0 = k.clone()
    policy = KIVI(bits=4, group_size=32, residual=128)
    k2, v2 = run(policy, k, v)
    assert policy.n_quant[0] == 64  # (200 - 128) // 32 * 32
    assert torch.equal(k2[:, :, 64:], k0[:, :, 64:])
    assert not torch.equal(k2[:, :, :64], k0[:, :, :64])
    full = 2 * k.numel() * k.element_size()
    assert policy.layer_logical_bytes(0, k2, v2) < full


def test_tiered_demotes_instead_of_evicting():
    k, v = random_kv(40)
    k0 = k.clone()
    policy = Tiered(budget=10, low_bits=4, sink=4, recent=0.2)  # recent = 2 tokens
    k2, _ = run(policy, k, v, torch.rand(B, H, 40))
    assert k2.shape[2] == 40                       # nothing removed
    low = policy.low[0][0, 0]
    assert int(low.sum()) == 30                    # 40 - budget demoted
    assert not low[:4].any() and not low[-2:].any()  # sinks and recent stay full precision
    assert torch.equal(k2[0, 0, ~low], k0[0, 0, ~low])


def test_pyramid_budgets_keep_the_average():
    b = budgets.pyramid(100, 16, 0.5)
    assert b[0] > b[-1] and abs(sum(b) - 1600) <= 16


def test_adaptive_budgets_favour_diffuse_layers():
    flat = torch.ones(1, 2, 500)
    peaked = torch.full((1, 2, 500), 1e-4)
    peaked[..., :5] = 1.0
    b = budgets.adaptive(100, {0: flat, 1: peaked}, 2)
    assert b[0] > b[1] and abs(sum(b) - 200) <= 2


def test_build_from_config():
    p = build({"type": "h2o", "recent": 0.5}, budget=64)
    assert isinstance(p, H2O) and p.avg_budget == 64
    assert isinstance(build({"type": "kivi", "bits": 2}), KIVI)


def test_invalid_budget_is_rejected():
    with pytest.raises(ValueError):
        StreamingLLM(budget=4, sink=4)
