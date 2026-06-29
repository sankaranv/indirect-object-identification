import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    from analysis import (
        attention_to_positions,
        ov_copy_strength,
        unembed_projections_at_positions,
    )

    assert callable(attention_to_positions)
    assert callable(unembed_projections_at_positions)
    assert callable(ov_copy_strength)


def test_attention_to_positions_returns_dict():
    import torch

    from analysis import attention_to_positions
    from model import load_model

    model = load_model()
    N, seq = 2, 10
    tokens = torch.randint(0, 1000, (N, seq))
    query_pos = torch.tensor([seq - 1, seq - 1])
    key_pos = torch.tensor([0, 0])
    result = attention_to_positions(model, tokens, query_pos, key_pos)
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    assert len(result) == n_layers * n_heads
    assert all(isinstance(v, float) for v in result.values())


def test_unembed_projections_shape():
    import torch

    from analysis import unembed_projections_at_positions
    from model import load_model

    model = load_model()
    N, seq = 2, 10
    tokens = torch.randint(0, 1000, (N, seq))
    positions = torch.tensor([seq - 1, seq - 1])
    proj = unembed_projections_at_positions(model, tokens, positions)
    n_layers = len(model.transformer.h)
    vocab = model.config.vocab_size
    assert proj.shape == (n_layers, N, vocab)
