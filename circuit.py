from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch

CIRCUIT: Dict[str, List[Tuple[int, int]]] = {
    "name_mover": [(9, 9), (10, 0), (9, 6)],
    "backup_name_mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "negative_name_mover": [(10, 7), (11, 10)],
    "s2_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
}

# companion sets for minimality: score(v) = |F(C\K) - F(C\K∪{v})|
K_FOR_EACH_COMPONENT: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {
    (9, 9): set(),
    (10, 0): {(9, 9)},
    (9, 6): {(9, 9), (10, 0)},
    (10, 7): {(11, 10)},
    (11, 10): {(10, 7)},
    (8, 10): {(7, 9), (8, 6), (7, 3)},
    (7, 9): {(8, 10), (8, 6), (7, 3)},
    (8, 6): {(7, 9), (8, 10), (7, 3)},
    (7, 3): {(7, 9), (8, 10), (8, 6)},
    (5, 5): {(5, 9), (6, 9), (5, 8)},
    (5, 9): {(11, 10), (10, 7)},
    (6, 9): {(5, 9), (5, 5), (5, 8)},
    (5, 8): {(11, 10), (10, 7)},
    (0, 1): {(0, 10), (3, 0)},
    (0, 10): {(0, 1), (3, 0)},
    (3, 0): {(0, 1), (0, 10)},
    (4, 11): {(2, 2)},
    (2, 2): {(4, 11)},
    (11, 2): {(9, 9), (10, 0), (9, 6)},
    (10, 6): {(9, 9), (10, 0), (9, 6), (11, 2)},
    (10, 10): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6)},
    (10, 2): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10)},
    (9, 7): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2)},
    (10, 1): {(9, 9), (10, 0), (9, 6), (11, 2), (10, 6), (10, 10), (10, 2), (9, 7)},
    (11, 9): {(9, 9), (10, 0), (9, 6), (9, 0)},
    (9, 0): {(9, 9), (10, 0), (9, 6), (11, 9)},
}

SEQ_POS_TO_KEEP: Dict[str, str] = {
    "name_mover": "end",
    "backup_name_mover": "end",
    "negative_name_mover": "end",
    "s2_inhibition": "end",
    "induction": "S2",
    "duplicate_token": "S2",
    "previous_token": "S+1",
}


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
    # Per-template-group means capture the token-distribution shift from ABC corruption
    # (IO/S names replaced by random names) without contaminating the mean with the
    # original name tokens. This matches the mean-ablation methodology in
    # Wang et al. 2022.
    for layer in range(n_layers):
        with model.trace({"input_ids": corrupted_toks}):
            z = model.transformer.h[layer].attn.c_proj.input.save()
        z_heads = z.reshape(batch_size, seq_len, n_heads, d_head)
        for group in groups:
            group_mean = z_heads[group].mean(0)
            means[layer, group] = group_mean.cpu()
    return means


def run_with_mean_ablation(
    model,
    clean_toks: torch.Tensor,
    means: torch.Tensor,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
    word_idx: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run model replacing z with means except circuit heads at their relevant
    positions.

    Returns logits [N, seq, vocab].
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    batch_size, seq_len = clean_toks.shape

    keep: Dict[int, torch.Tensor] = {
        layer: torch.zeros(batch_size, seq_len, n_heads, dtype=torch.bool)
        for layer in range(n_layers)
    }
    for head_type, head_list in circuit.items():
        positions = word_idx[seq_pos_to_keep[head_type]]
        for layer, head in head_list:
            keep[layer][torch.arange(batch_size), positions, head] = True

    # Non-circuit heads are replaced with their per-template-group mean activation,
    # isolating the circuit's contribution while preserving the residual stream's
    # statistical structure. Circuit heads at their relevant sequence positions are
    # kept clean.
    with model.trace({"input_ids": clean_toks}):
        for layer in range(n_layers):
            z = model.transformer.h[layer].attn.c_proj.input
            z_h = z.reshape(batch_size, seq_len, n_heads, d_head)
            mask = keep[layer].unsqueeze(-1).to(z.device)
            z_new = torch.where(mask, z_h, means[layer].to(z.device))
            model.transformer.h[layer].attn.c_proj.input[:] = z_new.reshape(
                batch_size, seq_len, n_heads * d_head
            )
        logits = model.lm_head.output.save()
    return logits
