"""Appendix H+I / Fig 18: Duplicate, previous-token, and induction score heatmaps.

Three attention-pattern scores computed on AA repeated random-token sequences:
  - Duplicate score: mean attention from position i (second half) to i-seq_len
                     (i.e. to the first occurrence of the same token)
  - Previous-token score: mean attention from position i to position i-1
  - Induction score: mean attention from position i (second half) to i-seq_len+1
                     (i.e. to the token *after* the first occurrence)

Run from indirect-object-identification/:
    python experiments/appHI_dup_scores.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import clear_cache, load_model


def compute_all_scores(model, seq_len: int = 100, batch: int = 5) -> tuple:
    """Return (dup, prev, ind) dicts: {(layer, head): score}.

    Computed on AA repeated sequences of total length 2*seq_len.

    dup:  mean attn from pos i (second half) to pos i-seq_len (first half copy)
    prev: mean off-diagonal attn from pos i to pos i-1
    ind:  mean attn from pos i (second half) to pos i-seq_len+1
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    T = 2 * seq_len

    torch.manual_seed(42)
    half = torch.randint(1, model.config.vocab_size, (batch, seq_len))
    tokens = torch.cat([half, half], dim=1)  # [batch, T]

    dup = {(layer, head): 0.0 for layer in range(n_layers) for head in range(n_heads)}
    prev = {(layer, head): 0.0 for layer in range(n_layers) for head in range(n_heads)}
    ind = {(layer, head): 0.0 for layer in range(n_layers) for head in range(n_heads)}

    for layer in range(n_layers):
        # Trace with output_attentions=True to get per-head attention weights
        with model.trace({"input_ids": tokens}, output_attentions=True):
            w = model.transformer.h[layer].attn.output[1].save()
        # w shape: [batch, n_heads, T, T]
        w_cpu = w.cpu().float()
        del w
        clear_cache()

        for head in range(n_heads):
            attn = w_cpu[:, head, :, :]  # [batch, T, T]

            # Duplicate score: attn from pos i (i >= seq_len) to pos i-seq_len
            dup_score = 0.0
            for i in range(seq_len, T):
                dup_score += attn[:, i, i - seq_len].mean().item()
            dup[(layer, head)] = dup_score / seq_len

            # Previous-token score: mean of attn[i, i-1] for i in 1..T-1
            prev_score = 0.0
            for i in range(1, T):
                prev_score += attn[:, i, i - 1].mean().item()
            prev[(layer, head)] = prev_score / (T - 1)

            # Induction score: attn from pos i (i >= seq_len) to pos i-seq_len+1
            ind_score = 0.0
            n_ind = 0
            for i in range(seq_len, T):
                j = i - seq_len + 1
                if j < T:
                    ind_score += attn[:, i, j].mean().item()
                    n_ind += 1
            ind[(layer, head)] = ind_score / n_ind if n_ind > 0 else 0.0

    return dup, prev, ind


def run() -> None:
    torch.set_grad_enabled(False)
    model = load_model()
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head

    dup, prev, ind = compute_all_scores(model)

    os.makedirs("plots/copy_scores", exist_ok=True)

    for scores, fname, title in [
        (dup, "fig18_dup.png", "Duplicate token score (attention to first occurrence)"),
        (
            prev,
            "fig18_prev.png",
            "Previous token score (mean attention from pos i to i-1)",
        ),
        (
            ind,
            "fig18_ind.png",
            "Induction score (attention to token after first occurrence)",
        ),
    ]:
        arr = np.zeros((n_layers, n_heads))
        for (layer, head), v in scores.items():
            arr[layer, head] = v

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(arr, cmap="Blues", vmin=0)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(f"plots/copy_scores/{fname}", dpi=150)
        plt.close()
        print(f"Saved plots/copy_scores/{fname}")


if __name__ == "__main__":
    run()
