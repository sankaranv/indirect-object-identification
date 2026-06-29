import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch

from metrics import kl_divergence, logit_diff


def test_logit_diff_shape_and_value():
    N, V = 4, 100
    logits = torch.zeros(N, V)
    correct_ids = [5, 10, 15, 20]
    incorrect_ids = [6, 11, 16, 21]
    for i in range(N):
        logits[i, correct_ids[i]] = 2.0
        logits[i, incorrect_ids[i]] = 0.5
    diff = logit_diff(logits, correct_ids, incorrect_ids)
    assert diff.shape == (N,)
    assert torch.allclose(diff, torch.full((N,), 1.5))


def test_logit_diff_accepts_tensors():
    logits = torch.randn(3, 50)
    diff = logit_diff(logits, torch.tensor([0, 1, 2]), torch.tensor([3, 4, 5]))
    assert diff.shape == (3,)


def test_logit_diff_requires_2d():
    import pytest

    with pytest.raises(AssertionError):
        logit_diff(torch.randn(3, 10, 50), [0, 1, 2], [3, 4, 5])


def test_kl_divergence_shape():
    N, V = 5, 200
    logits = torch.randn(N, V)
    reference = torch.randn(N, V)
    result = kl_divergence(logits, reference)
    assert result.shape == (N,)


def test_kl_divergence_zero_when_identical():
    N, V = 4, 100
    logits = torch.randn(N, V)
    result = kl_divergence(logits, logits)
    assert torch.allclose(result, torch.zeros(N), atol=1e-5)


def test_kl_divergence_nonnegative():
    N, V = 6, 50
    logits = torch.randn(N, V)
    reference = torch.randn(N, V)
    assert (kl_divergence(logits, reference) >= 0).all()


def test_kl_divergence_requires_2d():
    import pytest

    with pytest.raises(AssertionError):
        kl_divergence(torch.randn(3, 10, 50), torch.randn(3, 10, 50))


def test_kl_divergence_shape_mismatch():
    import pytest

    with pytest.raises(AssertionError):
        kl_divergence(torch.randn(3, 50), torch.randn(4, 50))
