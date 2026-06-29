import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_head_output_io_projection_shape():
    """Shape test using a synthetic mock model — no real GPT-2 load."""
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    import torch

    from analysis import head_output_io_projection

    n_layers, n_heads, d_model = 2, 4, 8
    d_head = d_model // n_heads
    N, seq = 3, 10

    # Build a lightweight fake model without loading GPT-2.
    model = MagicMock()
    model.config.n_head = n_heads
    model.config.n_embd = d_model
    # lm_head.weight: [vocab, d_model]
    model.lm_head.weight = torch.randn(50, d_model)

    fake_layers = []
    for _ in range(n_layers):
        layer = MagicMock()
        # GPT-2 Conv1D convention: c_proj.weight shape is [n_heads*d_head, d_model]
        layer.attn.c_proj.weight = torch.randn(n_heads * d_head, d_model)
        # c_proj.input.save() returns the pre-projection activations:
        # [N, seq, n_heads*d_head]
        z_val = torch.randn(N, seq, n_heads * d_head)
        layer.attn.c_proj.input.save = MagicMock(return_value=z_val)
        fake_layers.append(layer)
    model.transformer.h = fake_layers

    @contextmanager
    def fake_trace(*args, **kwargs):
        yield

    model.trace = fake_trace

    tokens = torch.randint(0, 50, (N, seq))
    end_pos = torch.tensor([seq - 1] * N)
    io_ids = list(range(N))  # token IDs for IO names
    s_ids = list(range(N, 2 * N))  # token IDs for S names

    result = head_output_io_projection(model, tokens, end_pos, io_ids, s_ids)

    assert len(result) == n_layers * n_heads, (
        f"Expected {n_layers * n_heads} entries, got {len(result)}"
    )
    for (layer, head), v in result.items():
        assert v.shape == (N,), (
            f"head ({layer},{head}): expected shape ({N},), got {v.shape}"
        )
