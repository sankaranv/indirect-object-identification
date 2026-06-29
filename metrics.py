from __future__ import annotations

from typing import List, Union

import torch


def logit_diff(
    logits: torch.Tensor,
    correct_ids: Union[List[int], torch.Tensor],
    incorrect_ids: Union[List[int], torch.Tensor],
) -> torch.Tensor:
    """logit(correct) − logit(incorrect) per example.

    logits: [N, vocab] — caller selects the sequence position before calling.
    """
    assert logits.ndim == 2, f"expected [N, vocab], got shape {logits.shape}"
    rows = torch.arange(logits.size(0), device=logits.device)
    if not isinstance(correct_ids, torch.Tensor):
        correct_ids = torch.tensor(correct_ids, device=logits.device)
    if not isinstance(incorrect_ids, torch.Tensor):
        incorrect_ids = torch.tensor(incorrect_ids, device=logits.device)
    return (
        logits[rows, correct_ids.to(logits.device)]
        - logits[rows, incorrect_ids.to(logits.device)]
    )
