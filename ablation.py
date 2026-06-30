from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch

Ablation = Callable[[int], torch.Tensor]
# (layer_idx) -> Tensor[N, seq, n_heads, d_head]
# Returns the substitute activation for every (example, position, head) triple.


def compute_means(
    model,
    corrupted_toks: torch.Tensor,
    groups: List[List[int]],
) -> torch.Tensor:
    """Per-template-group mean z-vectors over the corrupted (ABC) dataset.

    Returns Tensor[n_layers, N, seq, n_heads, d_head].
    means[layer, i] is the template-group mean z for example i.
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    batch_size, seq_len = corrupted_toks.shape

    means = torch.zeros(n_layers, batch_size, seq_len, n_heads, d_head)
    # Per-template-group means capture the token-distribution shift from ABC
    # corruption without contaminating with the original name tokens, matching
    # the mean-ablation methodology in Wang et al. 2022.
    for layer in range(n_layers):
        with model.trace({"input_ids": corrupted_toks}):
            z = model.transformer.h[layer].attn.c_proj.input.save()
        z_heads = z.reshape(batch_size, seq_len, n_heads, d_head)
        for group in groups:
            group_mean = z_heads[group].mean(0)
            means[layer, group] = group_mean.cpu()
    return means


def mean_ablation(means: torch.Tensor) -> Ablation:
    """Substitute with per-template-group means from compute_means().

    means: Tensor[n_layers, N, seq, n_heads, d_head]
    """
    return lambda layer: means[layer]


def zero_ablation(model, toks: torch.Tensor) -> Ablation:
    """Substitute all activations with zeros."""
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    batch_size, seq_len = toks.shape
    zeros = torch.zeros(batch_size, seq_len, n_heads, d_head)
    return lambda _: zeros


def resample_ablation(
    model,
    corrupted_toks: torch.Tensor,
    seed: Optional[int] = None,
) -> Ablation:
    """Substitute example i's activation with a randomly resampled corrupted activation.

    A fixed random permutation is applied across examples so that the substitute
    is in-distribution but uncorrelated with the clean input. The permutation is
    fixed at factory time for reproducibility.
    """
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    batch_size, seq_len = corrupted_toks.shape

    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)
    perm = torch.randperm(batch_size, generator=gen)

    cached: Dict[int, torch.Tensor] = {}
    for layer in range(len(model.transformer.h)):
        with model.trace({"input_ids": corrupted_toks}):
            z = model.transformer.h[layer].attn.c_proj.input.save()
        cached[layer] = z.reshape(batch_size, seq_len, n_heads, d_head)[perm].cpu()

    return lambda layer: cached[layer]


def counterfactual_ablation(model, corrupted_toks: torch.Tensor) -> Ablation:
    """Substitute example i's activation with example i's paired corrupted activation.

    Unlike resample_ablation, no permutation is applied — the substitute is the
    structurally paired counterfactual (e.g., the ABC prompt matched to the IOI
    prompt). This is the recommended scheme when clean and corrupted prompts are
    paired 1:1.
    """
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    batch_size, seq_len = corrupted_toks.shape

    cached: Dict[int, torch.Tensor] = {}
    for layer in range(len(model.transformer.h)):
        with model.trace({"input_ids": corrupted_toks}):
            z = model.transformer.h[layer].attn.c_proj.input.save()
        cached[layer] = z.reshape(batch_size, seq_len, n_heads, d_head).cpu()

    return lambda layer: cached[layer]
