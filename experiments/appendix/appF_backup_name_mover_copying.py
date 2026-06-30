"""Appendix F / Fig 16: BNM scatter — attention-to-IO vs IO-direction projection.

For each of the four focal Backup Name Mover heads [(10,10), (10,2), (11,2), (9,7)],
produce a separate scatter plot where every point is one example:

  x-axis : attention probability from END position to IO token
  y-axis : head output at END projected onto W_U[IO] − W_U[S]

Heads that genuinely "back up" the Name Mover circuit should cluster in the
upper-right quadrant: they attend to IO *and* write in the IO direction.
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import N as N_EXAMPLES, SEED

from analysis import head_output_io_projection
from ioi_dataset import IOIDataset
from model import clear_cache, load_model

BNM_HEADS = [(10, 10), (10, 2), (11, 2), (9, 7)]
OUT_DIR = "plots/backup"


def get_per_example_attn(model, tokens, end_pos, io_pos, heads):
    """Return per-example attention probability from END to IO for each head.

    Runs one nnsight trace per unique layer to minimise forward passes.

    Returns: {(layer, head): Tensor[N]}
    """
    N = tokens.size(0)
    layers_needed = sorted(set(lh[0] for lh in heads))
    result = {}
    for layer in layers_needed:
        with model.trace({"input_ids": tokens}, output_attentions=True):
            w = model.transformer.h[layer].attn.output[1].save()
        # w: [N, n_heads, seq, seq]
        for lh in heads:
            if lh[0] == layer:
                head = lh[1]
                result[lh] = w[torch.arange(N), head, end_pos, io_pos].cpu()
        del w
        clear_cache()
    return result


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(SEED)
    np.random.seed(SEED)
    ioi = IOIDataset("mixed", N=N_EXAMPLES, tokenizer=model.tokenizer, prepend_bos=False)
    end_pos = ioi.word_idx["end"].long()
    io_pos = ioi.word_idx["IO"].long()  # [N]

    attn_per_example = get_per_example_attn(
        model, ioi.toks.long(), end_pos, io_pos, BNM_HEADS
    )

    proj_all = head_output_io_projection(
        model, ioi.toks.long(), end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    for lh in BNM_HEADS:
        layer, head = lh
        attn_vals = attn_per_example[lh].numpy()  # [N]
        proj_vals = proj_all[lh].numpy()  # [N]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(attn_vals, proj_vals, s=18, alpha=0.45, color="#64B5F6", zorder=3)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Attention prob. to IO at END position")
        ax.set_ylabel("Head output · (W_U[IO] − W_U[S])")
        ax.set_title(
            f"Backup NM Head {layer}.{head}: attention vs IO-direction projection"
        )
        plt.tight_layout()

        fname = os.path.join(OUT_DIR, f"fig16_{layer}.{head}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Saved {fname}")

        print(
            f"Head {layer}.{head}: "
            f"mean_attn={attn_vals.mean():.3f}  mean_proj={proj_vals.mean():.2f}"
        )


if __name__ == "__main__":
    run()
