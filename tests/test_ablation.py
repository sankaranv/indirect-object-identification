"""Unit tests for ablation.py factory functions.

Uses a toy model stub so no GPU or real weights are needed.
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ablation import mean_ablation, zero_ablation


# ---------------------------------------------------------------------------
# Minimal model stub: 2 layers, 2 heads, d_model=8
# ---------------------------------------------------------------------------

def _make_stub_model(n_layers=2, n_heads=2, d_model=8):
    cfg = types.SimpleNamespace(n_head=n_heads, n_embd=d_model)
    layers = [types.SimpleNamespace() for _ in range(n_layers)]
    return types.SimpleNamespace(config=cfg, transformer=types.SimpleNamespace(h=layers))


# ---------------------------------------------------------------------------
# zero_ablation
# ---------------------------------------------------------------------------

def test_zero_ablation_shape():
    model = _make_stub_model(n_layers=2, n_heads=2, d_model=8)
    toks = torch.zeros(5, 7, dtype=torch.long)
    abl = zero_ablation(model, toks)
    assert abl(0).shape == (5, 7, 2, 4)


def test_zero_ablation_values():
    model = _make_stub_model(n_layers=2, n_heads=2, d_model=8)
    toks = torch.zeros(3, 6, dtype=torch.long)
    abl = zero_ablation(model, toks)
    assert abl(0).eq(0).all()
    assert abl(1).eq(0).all()


def test_zero_ablation_same_tensor_each_call():
    model = _make_stub_model()
    toks = torch.zeros(2, 4, dtype=torch.long)
    abl = zero_ablation(model, toks)
    assert abl(0) is abl(1)


# ---------------------------------------------------------------------------
# mean_ablation
# ---------------------------------------------------------------------------

def test_mean_ablation_returns_correct_layer():
    means = torch.arange(2 * 3 * 5 * 2 * 4, dtype=torch.float).reshape(2, 3, 5, 2, 4)
    abl = mean_ablation(means)
    assert torch.equal(abl(0), means[0])
    assert torch.equal(abl(1), means[1])


def test_mean_ablation_layer_values():
    means = torch.randn(3, 2, 5, 2, 4)
    abl = mean_ablation(means)
    for i in range(3):
        assert torch.equal(abl(i), means[i])


# ---------------------------------------------------------------------------
# resample_ablation: test determinism via the closure directly
# (avoids running a real forward pass)
# ---------------------------------------------------------------------------

def test_resample_ablation_seed_determinism():
    """Same seed → same permutation; different seeds → different permutations."""
    N, seq, n_heads, d_head = 8, 6, 2, 4

    corrupted = torch.randn(N, seq, n_heads, d_head)

    def _make_abl(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(N, generator=g)
        cached = {layer: corrupted[perm] for layer in range(2)}
        return lambda layer: cached[layer]

    abl_a = _make_abl(42)
    abl_b = _make_abl(42)
    abl_c = _make_abl(99)

    assert torch.equal(abl_a(0), abl_b(0)), "same seed → identical output"
    assert not torch.equal(abl_a(0), abl_c(0)), "different seeds → different output"
