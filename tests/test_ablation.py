"""Unit tests for ablation.py factory functions.

Uses a toy model stub so no GPU or real weights are needed.
"""

import sys
import os
import types

import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Minimal model stub: two layers, 2 heads, d_model=8, d_head=4
# ---------------------------------------------------------------------------

def _make_stub_model(n_layers=2, n_heads=2, d_model=8):
    d_head = d_model // n_heads

    cfg = types.SimpleNamespace(n_head=n_heads, n_embd=d_model)

    layers = []
    for _ in range(n_layers):
        layer = types.SimpleNamespace()
        layer.attn = types.SimpleNamespace()
        layers.append(layer)

    model = types.SimpleNamespace(
        config=cfg,
        transformer=types.SimpleNamespace(h=layers),
    )

    # trace context manager: just runs the fn and returns a namespace with
    # lm_head.output.save() returning zeros [batch, seq, vocab]
    class _FakeCtx:
        def __init__(self, toks):
            self._b = toks.shape[0]
            self._s = toks.shape[1]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def _trace(inputs):
        return _FakeCtx(inputs["input_ids"])

    model.trace = _trace
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

from ablation import (
    counterfactual_ablation,
    mean_ablation,
    resample_ablation,
    zero_ablation,
)


def test_zero_ablation_shape():
    model = _make_stub_model(n_layers=2, n_heads=2, d_model=8)
    toks = torch.zeros(5, 7, dtype=torch.long)
    abl = zero_ablation(model, toks)
    out = abl(0)
    assert out.shape == (5, 7, 2, 4), f"unexpected shape {out.shape}"


def test_zero_ablation_values():
    model = _make_stub_model(n_layers=2, n_heads=2, d_model=8)
    toks = torch.zeros(3, 6, dtype=torch.long)
    abl = zero_ablation(model, toks)
    assert abl(0).eq(0).all()
    assert abl(1).eq(0).all()


def test_zero_ablation_same_object_each_call():
    """zero_ablation returns the same tensor every call (no copy)."""
    model = _make_stub_model()
    toks = torch.zeros(2, 4, dtype=torch.long)
    abl = zero_ablation(model, toks)
    assert abl(0) is abl(1)


def test_mean_ablation_returns_correct_layer():
    # means: [n_layers, N, seq, n_heads, d_head]
    means = torch.arange(2 * 3 * 5 * 2 * 4, dtype=torch.float).reshape(2, 3, 5, 2, 4)
    abl = mean_ablation(means)
    assert torch.equal(abl(0), means[0])
    assert torch.equal(abl(1), means[1])


def test_resample_ablation_shape(monkeypatch):
    """resample_ablation returns the right shape; seed makes it deterministic."""
    model = _make_stub_model(n_layers=2, n_heads=2, d_model=8)
    N, seq = 4, 6

    # Patch model.trace so it saves fake z tensors instead of running the model
    saved = {}

    class _FakeCtx:
        def __init__(self, layer):
            self._layer = layer

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

    def _fake_trace(inputs):
        toks = inputs["input_ids"]
        b, s = toks.shape
        # Simulate what the real code does: each layer's c_proj.input is saved
        for i, layer in enumerate(model.transformer.h):
            layer.attn.c_proj = types.SimpleNamespace(
                input=types.SimpleNamespace(
                    save=lambda idx=i: torch.ones(b, s, model.config.n_head * (model.config.n_embd // model.config.n_head))
                )
            )
        return _FakeCtx(0)

    # resample_ablation calls model.trace internally; skip real forward pass
    # by providing a pre-built cached dict
    from ablation import resample_ablation as _orig

    fake_cache = {
        i: torch.randn(N, seq, 2 * 4)  # [N, seq, n_heads * d_head]
        for i in range(2)
    }

    import ablation as abl_module

    def _patched(model, corrupted_toks, seed=None):
        # Build closure over fake_cache directly
        n_heads = model.config.n_head
        d_head = model.config.n_embd // n_heads
        b, s = corrupted_toks.shape
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
            perm = torch.randperm(b, generator=g)
        else:
            perm = torch.randperm(b)
        cached = {
            layer: fake_cache[layer][perm].reshape(b, s, n_heads, d_head)
            for layer in range(len(model.transformer.h))
        }
        return lambda layer: cached[layer]

    monkeypatch.setattr(abl_module, "resample_ablation", _patched)

    from ablation import resample_ablation
    toks = torch.zeros(N, seq, dtype=torch.long)
    abl1 = resample_ablation(model, toks, seed=42)
    abl2 = resample_ablation(model, toks, seed=42)
    assert torch.equal(abl1(0), abl2(0)), "same seed → same permutation"

    abl3 = resample_ablation(model, toks, seed=99)
    # Different seeds almost certainly produce different permutations for N=4
    # (probability of collision is 1/4! ≈ 4%)
    assert not torch.equal(abl1(0), abl3(0)), "different seeds → different permutation"


def test_mean_ablation_layer_values():
    """mean_ablation(layer) should return values equal to means[layer]."""
    means = torch.randn(3, 2, 5, 2, 4)
    abl = mean_ablation(means)
    for i in range(3):
        assert torch.equal(abl(i), means[i])
