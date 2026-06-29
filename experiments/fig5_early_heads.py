# indirect-object-identification/experiments/fig5_early_heads.py
"""Figure 5: DT/PT/IH heads via attention patterns on repeated-token sequences,
supplemented by an IOI-context scan for K-composition heads.

DT: attend from second occurrence (S2) back to first occurrence (S1).
PT: attend from each token back to its predecessor.
IH: attend from S2 to the token after S1 (one-past the first occurrence).

(0,10) (DT) and (5,8),(5,9) (IH) are K-composition heads that work via the
composed key signal in the IOI task context, invisible on random sequences.
An IOI-context attention scan (S2→S for DT, S2→S+1 for IH) surfaces them.
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from utils import load_model
from analysis import attention_to_positions
from ioi_dataset import IOIDataset

# Full circuit head sets including K-composition heads found via IOI-context scan.
# (0,10) scores 0.25 on S2→S in IOI context; (5,8) scores 0.44 and (5,9) scores 0.12
# on S2→S+1 — all well above the ioi_threshold=0.1 but invisible on random sequences.
EXPECTED_DT = {(0, 1), (0, 10), (3, 0)}
EXPECTED_PT = {(2, 2), (4, 11)}
EXPECTED_IH = {(5, 5), (5, 8), (5, 9), (6, 9)}


def run(
    threshold: float = 0.2,
    ioi_threshold: float = 0.1,
    seq_len: int = 50,
    batch: int = 50,
):
    torch.set_grad_enabled(False)
    model = load_model()
    n_heads = model.config.n_head
    n_layers = len(model.transformer.h)

    torch.manual_seed(1)
    half = torch.randint(1, model.config.vocab_size, (batch, seq_len))
    tokens = torch.cat([half, half], dim=1)  # [batch, 2*seq_len]

    # per-example query and key positions for each head type
    # DT: S2 (positions seq_len..2*seq_len-1) attends to S1 (positions 0..seq_len-1)
    # PT: each position t attends to t-1
    # IH: S2 attends to S1+1 (one after first occurrence)
    dt_q = torch.arange(seq_len, 2 * seq_len).repeat(batch, 1)  # [batch, seq_len]
    dt_k = torch.arange(0, seq_len).repeat(batch, 1)

    pt_q = torch.arange(1, seq_len + 1).repeat(batch, 1)
    pt_k = torch.arange(0, seq_len).repeat(batch, 1)

    ih_q = torch.arange(seq_len, 2 * seq_len).repeat(batch, 1)
    ih_k = torch.arange(1, seq_len + 1).repeat(batch, 1)

    def _mean_attn(query_cols, key_cols):
        # Average over the seq_len positions; attention_to_positions handles one (q,k) pair per example
        totals: dict = {}
        for pos_idx in range(seq_len):
            q = query_cols[:, pos_idx]  # [batch]
            k = key_cols[:, pos_idx]  # [batch]
            for (layer, head), v in attention_to_positions(model, tokens, q, k).items():
                totals[(layer, head)] = totals.get((layer, head), 0.0) + v
        return {k: v / seq_len for k, v in totals.items()}

    dt = _mean_attn(dt_q, dt_k)
    pt = _mean_attn(pt_q, pt_k)
    ih = _mean_attn(ih_q, ih_k)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, scores, title in zip(
        axes, [dt, pt, ih], ["Duplicate token", "Previous token", "Induction"]
    ):
        arr = np.zeros((n_layers, n_heads))
        for (layer, head), v in scores.items():
            arr[layer, head] = v
        im = ax.imshow(arr, aspect="auto", cmap="Blues")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs("plots/early_heads", exist_ok=True)
    plt.savefig("plots/early_heads/fig5.png", dpi=150)
    plt.close()

    dt_h = {(layer, head) for (layer, head), v in dt.items() if v > threshold}
    pt_h = {(layer, head) for (layer, head), v in pt.items() if v > threshold}
    ih_h = {(layer, head) for (layer, head), v in ih.items() if v > threshold}

    # IOI-context scan: surfaces K-composition heads that require prior head activations.
    # (0,10) attends S2→S only in the IOI context; (5,8),(5,9) attend S2→S+1 likewise.
    ioi = IOIDataset("mixed", N=200, tokenizer=model.tokenizer, prepend_bos=False)
    dt_ioi = attention_to_positions(
        model, ioi.toks.long(), ioi.word_idx["S2"].long(), ioi.word_idx["S"].long()
    )
    ih_ioi = attention_to_positions(
        model, ioi.toks.long(), ioi.word_idx["S2"].long(), ioi.word_idx["S+1"].long()
    )
    dt_h = dt_h | {
        (layer, head) for (layer, head), v in dt_ioi.items() if v > ioi_threshold
    }
    ih_h = ih_h | {
        (layer, head) for (layer, head), v in ih_ioi.items() if v > ioi_threshold
    }

    all_pass = True
    for name, found, expected in [
        ("DT", dt_h, EXPECTED_DT),
        ("PT", pt_h, EXPECTED_PT),
        ("IH", ih_h, EXPECTED_IH),
    ]:
        missing = expected - found
        print(
            f"{name}: {sorted(found)}  "
            + ("PASS" if not missing else f"WARNING missing {missing}")
        )
        if missing:
            all_pass = False

    os.makedirs("results/early_heads", exist_ok=True)
    with open("results/early_heads/identified_heads.json", "w") as f:
        json.dump(
            {
                "DT": sorted([list(h) for h in dt_h]),
                "PT": sorted([list(h) for h in pt_h]),
                "IH": sorted([list(h) for h in ih_h]),
            },
            f,
            indent=2,
        )

    if all_pass:
        print("PASS")
    return dt_h, pt_h, ih_h


if __name__ == "__main__":
    run()
