"""Appendix C / Fig 12b: Path patching to S-Inhibition head keys at S2.

Which sender heads causally affect the KEY inputs of S-Inhibition heads?
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

from utils import load_model
from metrics import logit_diff
from patching import path_patch_head_to_heads
from ioi_dataset import IOIDataset

SI_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))

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
        receiver_heads=SI_HEADS,
        receiver_input="k",
        metric=metric,
    )

    si_min = min(layer for layer, _ in SI_HEADS)
    scores = {k: v for k, v in result.scores.items() if k[0] < si_min}

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
    ax.set_title("Head → S-Inhibition head keys causal effect at S2")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    os.makedirs("plots/si_keys", exist_ok=True)
    plt.savefig("plots/si_keys/fig12b.png", dpi=150)
    plt.close()
    print("Saved plots/si_keys/fig12b.png")

    return result


if __name__ == "__main__":
    run()
