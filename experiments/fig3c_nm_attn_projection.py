"""Figure 3c: scatter of attention-to-IO vs IO-direction projection for Name Mover heads.

Each point in the scatter is one (head, example) pair.  The three Name Mover heads
(9.9, 10.0, 9.6) are shown in distinct colours/markers.

x-axis : attention probability from the END position to the IO token
y-axis : head output at END projected onto W_U[IO] − W_U[S]

Heads that attend to IO and copy it should cluster in the upper-right quadrant.
"""
import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from utils import load_model, clear_cache
from analysis import head_output_io_projection
from ioi_dataset import IOIDataset

NM_HEADS = [(9, 9), (10, 0), (9, 6)]
COLORS   = {(9, 9): "#1f77b4", (10, 0): "#ff7f0e", (9, 6): "#2ca02c"}
MARKERS  = {(9, 9): "o",       (10, 0): "s",        (9, 6): "^"}


def get_per_example_attn(model, tokens, end_pos, io_pos, nm_heads):
    """Return per-example attention probability from END to IO for each NM head.

    Runs one nnsight trace per unique layer to minimise forward passes.

    Returns: {(layer, head): Tensor[N]}
    """
    N = tokens.size(0)
    layers_needed = sorted(set(lh[0] for lh in nm_heads))
    result = {}
    for layer in layers_needed:
        with model.trace({"input_ids": tokens}, output_attentions=True):
            w = model.transformer.h[layer].attn.output[1].save()
        # w: [N, n_heads, seq, seq]
        for lh in nm_heads:
            if lh[0] == layer:
                head = lh[1]
                result[lh] = w[torch.arange(N), head, end_pos, io_pos].cpu()
        del w
        clear_cache()
    return result


def run():
    model = load_model()
    random.seed(1)
    np.random.seed(1)
    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    N       = len(ioi)
    end_pos = ioi.word_idx["end"].long()   # [N]
    io_pos  = ioi.word_idx["IO"].long()    # [N]

    print(f"Dataset: {N} examples")
    print("Computing per-example attention probabilities …")
    attn_per_example = get_per_example_attn(
        model, ioi.toks.long(), end_pos, io_pos, NM_HEADS
    )

    print("Computing per-example IO-direction projections …")
    proj_all = head_output_io_projection(
        model, ioi.toks.long(), end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    for lh in NM_HEADS:
        attn_vals = attn_per_example[lh].numpy()   # [N]
        proj_vals = proj_all[lh].numpy()           # [N]
        ax.scatter(
            attn_vals,
            proj_vals,
            color=COLORS[lh],
            marker=MARKERS[lh],
            s=20,
            alpha=0.5,
            zorder=3,
            label=f"Head {lh[0]}.{lh[1]}",
        )

    ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Attention prob. to IO at END position")
    ax.set_ylabel("Head output · (W_U[IO] − W_U[S])")
    ax.set_title("Name Mover Heads: attention vs IO-direction projection")
    ax.legend()
    plt.tight_layout()

    out_dir = "plots/name_movers"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig3c.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # Summary statistics
    for lh in NM_HEADS:
        a = attn_per_example[lh]
        p = proj_all[lh]
        print(
            f"Head {lh[0]}.{lh[1]}: "
            f"mean_attn={a.mean():.3f}  mean_proj={p.mean():.2f}"
        )


if __name__ == "__main__":
    run()
