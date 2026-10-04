"""Checks of the analytic model against hand-computed / published numbers."""

import pytest

from kvlab import memory_model as mm


def test_kv_bytes_per_token_matches_hand_calculation():
    assert mm.kv_bytes_per_token(mm.LLAMA_32_1B) == 2 * 16 * 8 * 64 * 2 == 32 * 1024
    assert mm.kv_bytes_per_token(mm.LLAMA_31_8B) == 2 * 32 * 8 * 128 * 2 == 128 * 1024


def test_param_counts_match_published_sizes():
    assert mm.LLAMA_32_1B.total_params == pytest.approx(1.236e9, rel=0.01)
    assert mm.LLAMA_31_8B.total_params == pytest.approx(8.03e9, rel=0.01)


def test_quantization_overhead_is_counted():
    assert mm.kv_bits_per_element(16) == 16
    assert mm.kv_bits_per_element(4, 32) == 5  # 4 bits + (16-bit scale + 16-bit zero) / 32


def test_gqa_is_4x_smaller_than_mha():
    m = mm.LLAMA_31_8B
    assert mm.kv_bytes_per_token(m.with_kv_heads(m.heads)) == 4 * mm.kv_bytes_per_token(m)


def test_budget_caps_cache_size():
    m = mm.LLAMA_32_1B
    assert mm.kv_cache_bytes(m, 100_000, budget=1024) == mm.kv_cache_bytes(m, 1024)


def test_decode_is_memory_bound_at_batch_1():
    r = mm.decode_step_roofline(mm.LLAMA_32_1B, mm.GPUS["A100-80GB"], context=4096)
    assert r["bound"] == "memory"
    assert r["intensity_flops_per_byte"] < r["gpu_balance_flops_per_byte"]


def test_recomputing_costs_more_than_caching():
    assert mm.recompute_vs_cache(mm.LLAMA_32_1B, 1024, 64)["compute_ratio"] > 10
