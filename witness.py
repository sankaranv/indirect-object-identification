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

import torch
from ablation import counterfactual_ablation
from metrics import Metric
from model import clear_cache
from patching import PatchingResult, _batches


def witness_pinned_ablation_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    witness_heads: list[tuple[int, int]],
    *,
    suspect_heads: list[tuple[int, int]] | None = None,
    positions: torch.Tensor | None = None,
    batch_size: int | None = None,
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

    batch_effects: dict[tuple[int, int], list[float]] = {s: [] for s in suspects}

    ablation = counterfactual_ablation(model, corrupted)

    # Index witnesses by layer so the trace can write in ascending order.
    # nnsight 0.7 requires all interventions on c_proj.input to be registered
    # in ascending layer order within a single trace call.
    witness_heads_by_layer: dict[int, list[int]] = {}
    for wl, wh in witness_heads:
        witness_heads_by_layer.setdefault(wl, []).append(wh)

    for batch_start, batch_end, clean_b, _ in _batches(clean, corrupted, batch_size):
        batch_n = clean_b.shape[0]
        pos_b = (
            positions[batch_start:batch_end].long() if positions is not None else None
        )

        # Cache clean z for witness pinning — factual activations the witnesses
        # would have produced if the suspect had not been touched.
        clean_z: dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": clean_b}):
            for layer in range(n_layers):
                clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()

        clean_m = metric(clean_logits.cpu())
        # Device of the saved activations (MPS on Apple Silicon, CUDA on GPU).
        # counterfactual_ablation caches to CPU; move to model device before trace writes.
        model_device = clean_z[0].device

        for suspect_layer, suspect_head in suspects:
            suspect_slice = slice(suspect_head * d_head, (suspect_head + 1) * d_head)
            # Counterfactual value for the suspect head from the ablation.
            # ablation(layer) returns [N, seq, n_heads, d_head]; select this head.
            cf_head_z = ablation(suspect_layer)[
                batch_start:batch_end, :, suspect_head, :
            ].to(model_device)  # [batch_n, seq, d_head]

            with model.trace({"input_ids": clean_b}):
                # All writes in ascending layer order (nnsight 0.7 requirement).
                for layer in range(n_layers):
                    if layer == suspect_layer:
                        if pos_b is None:
                            suspect_z = cf_head_z
                        else:
                            # CF only at end_pos; factual (clean) z at all other
                            # positions for this head. Slice write leaves sibling
                            # heads at suspect_layer live — they recompute from the
                            # clean residual stream.
                            suspect_z = clean_z[layer][..., suspect_slice].clone()
                            batch_idx = torch.arange(batch_n)
                            suspect_z[batch_idx, pos_b] = cf_head_z[batch_idx, pos_b]
                        model.transformer.h[layer].attn.c_proj.input[
                            ..., suspect_slice
                        ] = suspect_z
                    # Pin witnesses at this layer to their factual (clean) activations.
                    for wh in witness_heads_by_layer.get(layer, []):
                        wslice = slice(wh * d_head, (wh + 1) * d_head)
                        model.transformer.h[layer].attn.c_proj.input[..., wslice] = (
                            clean_z[layer][..., wslice]
                        )
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            batch_effects[(suspect_layer, suspect_head)].append(
                (patched_m - clean_m).mean().item()
            )
            del patched_logits
            clear_cache()

        del clean_z, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)


def witness_importance_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    suspect_head: tuple[int, int],
    candidate_witnesses: list[tuple[int, int]],
    *,
    positions: torch.Tensor | None = None,
    batch_size: int | None = None,
) -> dict[tuple[int, int], float]:
    """Score each candidate witness by how much it suppresses backup compensation.

    For each candidate witness w, computes:
      importance(w) = |score with w pinned| − |score without w|

    High importance identifies heads that compensate when the suspect is absent.
    The baseline (no witnesses pinned) is computed once and reused.
    """
    # Baseline: suspect ablated, no witnesses — backup compensates freely.
    baseline_result = witness_pinned_ablation_scores(
        model,
        clean,
        corrupted,
        metric,
        witness_heads=[],
        suspect_heads=[suspect_head],
        positions=positions,
        batch_size=batch_size,
    )
    baseline_score = baseline_result.scores[suspect_head]

    # Exclude the suspect itself from the candidate list. If it were included,
    # the inner trace would first write CF to the suspect's slice (ablation) and
    # then immediately overwrite it with clean z (witness pin), silently undoing
    # the ablation and producing importance ≈ 0 for the suspect head.
    candidates = [c for c in candidate_witnesses if c != suspect_head]

    importance: dict[tuple[int, int], float] = {}
    for candidate in candidates:
        pinned_result = witness_pinned_ablation_scores(
            model,
            clean,
            corrupted,
            metric,
            witness_heads=[candidate],
            suspect_heads=[suspect_head],
            positions=positions,
            batch_size=batch_size,
        )
        pinned_score = pinned_result.scores[suspect_head]
        # Both scores are negative for helpful suspect heads. A more negative
        # pinned_score means the suspect looks more necessary when w is pinned.
        importance[candidate] = abs(pinned_score) - abs(baseline_score)

    return importance


def pie_denoising_scores(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Metric,
    *,
    batch_size: int | None = None,
) -> PatchingResult:
    """Score each head by how much restoring it alone improves a corrupted run.

    For each head h: runs the model on the corrupted input, replaces h's z-slice
    with its clean-run value, and freezes all other heads at their corrupted values.
    Score = patched_metric − corrupted_metric (positive for heads that help).

    This is path patching with clean and corrupted swapped. It identifies
    heads whose clean activation carries recoverable task information even when
    the rest of the circuit is disrupted (PIE / ADDER-gate denoising).

    Limitation: finds overdetermination backup heads (active in clean pass) but
    not preemption backups (dormant in clean pass, Δz ≈ 0).
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    batch_effects: dict[tuple[int, int], list[float]] = {
        (layer, head): [] for layer in range(n_layers) for head in range(n_heads)
    }

    for _, _, clean_b, corr_b in _batches(clean, corrupted, batch_size):
        # Cache corrupted z (the reference world) and clean z (restoration source).
        corrupted_z: dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": corr_b}):
            for layer in range(n_layers):
                corrupted_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
            corrupted_logits = model.lm_head.output.save()

        clean_z: dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": clean_b}):
            for layer in range(n_layers):
                clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

        corrupted_m = metric(corrupted_logits.cpu())

        for restore_layer in range(n_layers):
            for restore_head in range(n_heads):
                restore_slice = slice(
                    restore_head * d_head, (restore_head + 1) * d_head
                )
                with model.trace({"input_ids": corr_b}):
                    for layer in range(n_layers):
                        if layer == restore_layer:
                            # Restore this head to its clean activation; keep all
                            # other heads at their corrupted activations.
                            model.transformer.h[layer].attn.c_proj.input[
                                ..., restore_slice
                            ] = clean_z[restore_layer][..., restore_slice]
                        else:
                            model.transformer.h[layer].attn.c_proj.input[...] = (
                                corrupted_z[layer]
                            )
                    patched_logits = model.lm_head.output.save()

                patched_m = metric(patched_logits.cpu())
                batch_effects[(restore_layer, restore_head)].append(
                    (patched_m - corrupted_m).mean().item()
                )
                del patched_logits
                clear_cache()

        del corrupted_z, clean_z, corrupted_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)
