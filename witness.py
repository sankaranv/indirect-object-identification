"""Witness-pinned ablation and PIE denoising for IOI circuit analysis.

Three estimands, each differing in which downstream nodes are pinned:

  witness_pinned_ablation_scores
    Ablates a suspect head with counterfactual values and pins a specified
    set of witness heads at their factual (clean) activations. All other
    heads recompute freely. This tests necessity of the suspect while
    preventing named witnesses from compensating.

    Empty witness_heads → plain counterfactual NIE (backup compensates freely).
    All downstream heads as witnesses → approaches the path-patching estimand.

  witness_importance_scores
    For a single suspect head, scans candidate witnesses one at a time.
    Importance of witness w = |score with w pinned| − |score without w|.
    High importance identifies heads that compensate when the suspect is absent.

  pie_denoising_scores
    In a fully corrupted (ABC) run, restores one head at a time to its clean
    activation while pinning all other heads at corrupted values.
    Finds heads whose clean activation can single-handedly restore task
    performance in the absence of intact primary circuit components.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from ablation import counterfactual_ablation
from metrics import Metric
from patching import PatchingResult, _batches


def witness_pinned_ablation_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    witness_heads: List[Tuple[int, int]],
    *,
    suspect_heads: Optional[List[Tuple[int, int]]] = None,
    positions: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> PatchingResult:
    """Score each suspect head's necessity under counterfactual ablation with pinned witnesses.

    For each suspect head, replaces its z-slice with the paired counterfactual
    activation and simultaneously pins each witness head at its factual (clean)
    activation, preventing downstream compensation. All other heads recompute
    freely from the modified residual stream.

    suspect_heads: restrict scoring to this subset of heads; None scores all heads.
                   Pass [(layer, head)] when only one suspect's score is needed —
                   this is critical for performance in witness_importance_scores,
                   where calling with all 144 suspects but using only one result
                   would be ~144x slower than necessary.

    positions: per-example token index [N] at which the suspect is ablated;
               None ablates at every sequence position. Witnesses are always
               pinned at all positions (their full z-slice), which is the
               conservative choice: it prevents the witness from adapting to
               the suspect's modification at any position, not just the ablated one.

    Score sign matches Wang et al. / path_patch_head_to_logits convention:
    patched_metric − clean_metric, negative for heads that help the task.
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    suspects = (
        suspect_heads
        if suspect_heads is not None
        else [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    )

    batch_effects: Dict[Tuple[int, int], List[float]] = {s: [] for s in suspects}

    ablation = counterfactual_ablation(model, corrupted)

    for batch_start, batch_end, clean_b, corr_b in _batches(
        clean, corrupted, batch_size
    ):
        batch_n = clean_b.shape[0]
        pos_b = (
            positions[batch_start:batch_end].long() if positions is not None else None
        )

        # Cache clean z for witness pinning — factual activations the witnesses
        # would have produced if the suspect had not been touched.
        clean_z: Dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": clean_b}):
            for layer in range(n_layers):
                clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()

        clean_m = metric(clean_logits.cpu())

        for suspect_layer, suspect_head in suspects:
            suspect_slice = slice(suspect_head * d_head, (suspect_head + 1) * d_head)
            # Counterfactual value for the suspect head from the ablation.
            # ablation(layer) returns [N, seq, n_heads, d_head]; select this head.
            cf_head_z = ablation(suspect_layer)[
                batch_start:batch_end, :, suspect_head, :
            ]  # [batch_n, seq, d_head]

            with model.trace({"input_ids": clean_b}):
                # Ablate suspect at specified positions (or all positions).
                if pos_b is None:
                    model.transformer.h[suspect_layer].attn.c_proj.input[
                        ..., suspect_slice
                    ] = cf_head_z
                else:
                    # Precompute replacement: clean z everywhere, CF at pos_b.
                    # This is the safe pattern for position-restricted writes in
                    # nnsight — we cannot do indexed writes to proxy tensors, so
                    # we build the full replacement tensor outside the trace.
                    repl = clean_z[suspect_layer].clone()
                    batch_idx = torch.arange(batch_n)
                    repl[batch_idx, pos_b, suspect_slice] = cf_head_z[batch_idx, pos_b]
                    model.transformer.h[suspect_layer].attn.c_proj.input[...] = repl

                # Pin each witness at its factual (clean) activation at all
                # positions. This prevents the witness from recomputing in
                # response to the suspect's modified residual stream.
                for witness_layer, witness_head in witness_heads:
                    witness_slice = slice(
                        witness_head * d_head, (witness_head + 1) * d_head
                    )
                    model.transformer.h[witness_layer].attn.c_proj.input[
                        ..., witness_slice
                    ] = clean_z[witness_layer][..., witness_slice]

                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            batch_effects[(suspect_layer, suspect_head)].append(
                (patched_m - clean_m).mean().item()
            )
            del patched_logits
            torch.cuda.empty_cache()

        del clean_z, clean_logits
        torch.cuda.empty_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)


def witness_importance_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    suspect_head: Tuple[int, int],
    candidate_witnesses: List[Tuple[int, int]],
    *,
    positions: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """Score each candidate witness by how much it suppresses backup compensation.

    For each candidate witness w, computes:
      importance(w) = |score with w pinned| − |score without w|

    High importance identifies heads that compensate when the suspect is absent.
    The baseline (no witnesses pinned) is computed once and reused.
    """
    raise NotImplementedError("witness_importance_scores is implemented in Task 2")


def pie_denoising_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    *,
    batch_size: Optional[int] = None,
) -> PatchingResult:
    """Score each head by its ability to restore task performance from a corrupted run.

    In a fully corrupted (ABC) run, restores one head at a time to its clean
    activation while pinning all other heads at corrupted values. Finds heads
    whose clean activation can single-handedly restore task performance in the
    absence of intact primary circuit components.
    """
    raise NotImplementedError("pie_denoising_scores is implemented in Task 3")
