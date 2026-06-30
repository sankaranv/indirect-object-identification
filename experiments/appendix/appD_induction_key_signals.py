"""Appendix D / Fig 13b: Path patching to Induction head keys at S1+1.

Which sender heads causally affect the KEY inputs of Induction heads?
Uses path_patch_head_to_heads with receiver_input="k"; metric is logit diff
at the END position (final token), matching the main patching experiments.
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import N, SEED

from ioi_dataset import IOIDataset
from metrics import logit_diff
from model import load_model
from patching import path_patch_head_to_heads

IH_HEADS = [(5, 5), (5, 8), (5, 9), (6, 9)]


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(SEED)
    np.random.seed(SEED)

    ioi = IOIDataset("mixed", N=N, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))

    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()

    def metric(logits):
        return logit_diff(
            logits[torch.arange(N), end_pos],
            ioi.io_tokenIDs,
            ioi.s_tokenIDs,
        )

    result = path_patch_head_to_heads(
        model,
        ioi.toks.long(),
        abc.toks.long(),
        receiver_heads=IH_HEADS,
        receiver_input="k",
        metric=metric,
    )

    ih_min = min(layer for layer, _ in IH_HEADS)
    scores = {k: v for k, v in result.scores.items() if k[0] < ih_min}

    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    arr = np.zeros((n_layers, n_heads))
    for (layer, head), v in scores.items():
        arr[layer, head] = v

    vmax = max(np.abs(arr).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Head → Induction head keys causal effect at S1+1")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs("plots/ih_keys", exist_ok=True)
    plt.savefig("plots/ih_keys/fig13b.png", dpi=150)
    plt.close()
    print("Saved plots/ih_keys/fig13b.png")

    return result


if __name__ == "__main__":
    run()
