"""Appendix H / Fig 17: Per-head copy scores on repeated (AA) sequences.

For each attention head we compute the dot product between:
  - the head's output in residual space: z_h @ W_O_h
  - the unembedding vector of the *next* token: W_U[tokens[:, i+1]]
averaged over positions in the second half of a repeated random-token sequence.

Heads with high copy scores (especially induction heads IH) tend to copy the
token that followed the previous occurrence of the current token.

Run from indirect-object-identification/:
    python experiments/appH_copy_scores.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils import clear_cache, load_model

# Circuit head categories (layer, head)
CIRCUIT_HEADS = {
    "IH": {(5, 5), (5, 8), (5, 9), (6, 9)},
    "NM": {(9, 9), (10, 0), (9, 6)},
    "BNM": {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)},
    "NNM": {(10, 7), (11, 10)},
}

CIRCUIT_COLORS = {
    "IH": "#E74C3C",  # red
    "NM": "#2ECC71",  # green
    "BNM": "#3498DB",  # blue
    "NNM": "#9B59B6",  # purple
    "other": "#95A5A6",  # grey
}


def _head_circuit_type(layer: int, head: int) -> str:
    for ctype, heads in CIRCUIT_HEADS.items():
        if (layer, head) in heads:
            return ctype
    return "other"


def compute_copy_scores(model, seq_len: int = 50, batch: int = 10) -> dict:
    """Return {(layer, head): copy_score} on AA repeated random-token sequences.

    Copy score = mean over second-half positions i of
        dot(out_h[:, i, :], W_U[tokens[:, i+1]])
    where out_h = z_h @ W_O_h is the head contribution to the residual stream.
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    T = 2 * seq_len

    torch.manual_seed(42)
    half = torch.randint(1, model.config.vocab_size, (batch, seq_len))
    tokens = torch.cat([half, half], dim=1)  # [batch, T]

    # Unembedding matrix  [vocab, d_model]
    W_U = model.lm_head.weight.detach().cpu().float()

    scores: dict = {}

    for layer in range(n_layers):
        # Output projection weights  [n_heads*d_head, d_model]
        W_O = model.transformer.h[layer].attn.c_proj.weight.detach().cpu().float()

        # Trace: save c_proj input (= concatenated per-head z vectors)
        with model.trace({"input_ids": tokens}):
            z = model.transformer.h[layer].attn.c_proj.input.save()

        z_cpu = z.cpu().float()  # [batch, T, n_heads*d_head]
        del z
        clear_cache()

        # next-token indices for positions in second half: [batch, seq_len-1]
        next_toks_second = tokens[:, seq_len + 1 : T]  # T-1 positions → seq_len-1

        for head in range(n_heads):
            z_h = z_cpu[:, :, head * d_head : (head + 1) * d_head]  # [batch, T, d_head]
            W_O_h = W_O[head * d_head : (head + 1) * d_head, :]  # [d_head, d_model]
            out_h = z_h @ W_O_h  # [batch, T, d_model]

            # Focus on second-half positions where induction pattern is active
            # Exclude last position (no next token)
            out_second = out_h[:, seq_len : T - 1, :]  # [batch, seq_len-1, d_model]

            # Copy score: dot with unembedding of next token, mean over batch and pos
            score = (out_second * W_U[next_toks_second]).sum(-1).mean().item()
            scores[(layer, head)] = score

    return scores


def run() -> None:
    torch.set_grad_enabled(False)
    model = load_model()

    scores = compute_copy_scores(model)

    # Sort by absolute copy score and take top 20
    sorted_heads = sorted(scores.keys(), key=lambda k: abs(scores[k]), reverse=True)
    top20 = sorted_heads[:20]

    labels = [f"L{layer}H{head}" for layer, head in top20]
    values = [scores[k] for k in top20]
    colors = [CIRCUIT_COLORS[_head_circuit_type(layer, head)] for layer, head in top20]

    os.makedirs("plots/copy_scores", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(top20)), values, color=colors)
    ax.set_xticks(range(len(top20)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Copy score (mean dot with next-token unembedding)")
    ax.set_title("Per-head copy scores on repeated sequences (top 20 by |score|)")

    # Legend
    from matplotlib.patches import Patch

    legend_elems = [
        Patch(facecolor=CIRCUIT_COLORS[ct], label=ct)
        for ct in ("IH", "NM", "BNM", "NNM", "other")
    ]
    ax.legend(handles=legend_elems, loc="upper right")

    plt.tight_layout()
    plt.savefig("plots/copy_scores/fig17.png", dpi=150)
    plt.close()
    print("Saved plots/copy_scores/fig17.png")

    # Print top 10 for inspection
    print("\nTop 10 heads by |copy_score|:")
    for layer, head in sorted_heads[:10]:
        ct = _head_circuit_type(layer, head)
        print(f"  L{layer}H{head:2d}  {scores[(layer, head)]:+.4f}  [{ct}]")


if __name__ == "__main__":
    run()
