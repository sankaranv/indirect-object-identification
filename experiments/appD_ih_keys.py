"""Appendix D / Fig 13b: Path patching to Induction head keys at S1+1.

Which sender heads causally affect the KEY inputs of Induction heads?
Uses path_patch_head_to_heads with receiver_input="k"; metric is logit diff
at the END position (final token), matching the main patching experiments.
"""
import os
import sys
import random

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from utils import load_model
from metrics import logit_diff
from patching import path_patch_head_to_heads
from ioi_dataset import IOIDataset

IH_HEADS = [(5, 5), (5, 8), (5, 9), (6, 9)]


def run():
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))

    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()

    metric = lambda logits: logit_diff(
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

    ih_min = min(l for l, _ in IH_HEADS)
    scores = {k: v for k, v in result.scores.items() if k[0] < ih_min}

    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    arr = np.zeros((n_layers, n_heads))
    for (l, h), v in scores.items():
        arr[l, h] = v

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
