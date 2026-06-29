from __future__ import annotations

from typing import Callable, List, Union

import torch
import torch.nn.functional as F

# Metric: takes full logits [N, seq, vocab], returns per-example scalar [N].
# Callers are responsible for position selection inside their closure.
Metric = Callable[[torch.Tensor], torch.Tensor]


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


def kl_divergence(
    logits: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """KL(p_reference ∥ p_logits) per example — how far logits drifts from reference.

    logits    : [N, vocab] — the distribution being evaluated (e.g. patched model).
    reference : [N, vocab] — the reference distribution (e.g. clean model logits).

    Uses the forward KL so that every token the reference puts mass on must be
    covered by logits; under-coverage is penalised.
    """
    assert logits.ndim == 2, f"expected [N, vocab], got shape {logits.shape}"
    assert reference.shape == logits.shape, (
        f"reference shape {reference.shape} != logits shape {logits.shape}"
    )
    log_p = F.log_softmax(logits.float(), dim=-1)
    p_ref = F.softmax(reference.float(), dim=-1)
    # F.kl_div(log_q, p) computes KL(p ∥ q); reduction="none" → [N, vocab]
    return F.kl_div(log_p, p_ref, reduction="none").sum(-1)
