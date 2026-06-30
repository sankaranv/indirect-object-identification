from __future__ import annotations

from typing import Dict, List, Set, Tuple

import torch

from ablation import Ablation

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


def run_with_ablation(
    model,
    clean_toks: torch.Tensor,
    ablation: Ablation,
    circuit: Dict[str, List[Tuple[int, int]]],
    seq_pos_to_keep: Dict[str, str],
    word_idx: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Run model with non-circuit head z replaced by ablation(layer).

    Circuit heads at their relevant sequence positions are kept clean.
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

    with model.trace({"input_ids": clean_toks}):
        for layer in range(n_layers):
            z = model.transformer.h[layer].attn.c_proj.input
            z_h = z.reshape(batch_size, seq_len, n_heads, d_head)
            mask = keep[layer].unsqueeze(-1).to(z.device)
            z_new = torch.where(mask, z_h, ablation(layer).to(z.device))
            model.transformer.h[layer].attn.c_proj.input[:] = z_new.reshape(
                batch_size, seq_len, n_heads * d_head
            )
        logits = model.lm_head.output.save()
    return logits
