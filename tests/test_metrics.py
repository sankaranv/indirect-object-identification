import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from metrics import logit_diff

def test_logit_diff_shape_and_value():
    N, V = 4, 100
    logits = torch.zeros(N, V)
    correct_ids   = [5, 10, 15, 20]
    incorrect_ids = [6, 11, 16, 21]
    for i in range(N):
        logits[i, correct_ids[i]]   = 2.0
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
