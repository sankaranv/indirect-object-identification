"""Bar chart of logit diff when patching each SI head individually and all four
together."""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import SEED

from ioi_dataset import IOIDataset
from metrics import logit_diff
from model import load_model

SI_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]


def _patch_heads_output(model, clean_toks, corrupted_toks, heads):
    """Activation patch: replace z output of `heads` from corrupted into clean run."""
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    # Group heads by layer
    patch_layers = {}
    for layer, head in heads:
        patch_layers.setdefault(layer, []).append(head)

    # Cache corrupted z per layer
    corr_z = {}
    with model.trace({"input_ids": corrupted_toks}):
        for layer in patch_layers:
            corr_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

    # Patch corrupted z for selected heads into clean run
    with model.trace({"input_ids": clean_toks}):
        for layer, head_list in patch_layers.items():
            for head in head_list:
                z_sl = slice(head * d_head, (head + 1) * d_head)
                model.transformer.h[layer].attn.c_proj.input[..., z_sl] = corr_z[layer][
                    ..., z_sl
                ]
        logits = model.lm_head.output.save()

    return logits.cpu()


def run():
    torch.set_grad_enabled(False)
    model = load_model()

    random.seed(SEED)
    np.random.seed(SEED)
    ioi = IOIDataset("mixed", N=100, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))

    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()

    def ld_per_example(logits):
        return (
            logit_diff(
                logits[torch.arange(N), end_pos],
                ioi.io_tokenIDs,
                ioi.s_tokenIDs,
            )
            .cpu()
            .numpy()
        )

    # Baseline: clean run, no patching
    with model.trace({"input_ids": ioi.toks.long()}):
        base_logits = model.lm_head.output.save()
    base_ld = ld_per_example(base_logits.cpu())

    # Patch each SI head individually
    conditions = {}
    for lh in SI_HEADS:
        logits = _patch_heads_output(model, ioi.toks.long(), abc.toks.long(), [lh])
        conditions[f"{lh[0]}.{lh[1]}"] = ld_per_example(logits)

    # Patch all 4 SI heads together
    logits_all = _patch_heads_output(model, ioi.toks.long(), abc.toks.long(), SI_HEADS)
    conditions["All 4 SI"] = ld_per_example(logits_all)

    labels = [f"{lh[0]}.{lh[1]}" for lh in SI_HEADS] + ["All 4 SI"]
    means = [conditions[k].mean() for k in labels]
    stds = [conditions[k].std() for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        range(len(labels)),
        means,
        yerr=stds,
        capsize=4,
        color=["#FF9800"] * 4 + ["#E65100"],
    )
    ax.axhline(
        base_ld.mean(),
        color="k",
        ls="--",
        lw=1,
        label=f"Baseline ({base_ld.mean():.2f})",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Logit diff (IO − S)")
    ax.set_title("Effect of patching S-Inhibition heads from ABC dataset")
    ax.legend()
    plt.tight_layout()

    os.makedirs("plots/s2_inhibition", exist_ok=True)
    plt.savefig("plots/s2_inhibition/fig4c.png", dpi=150)
    plt.close()

    os.makedirs("results/s2_inhibition", exist_ok=True)
    rows = [
        {
            "condition": k,
            "mean_logit_diff": conditions[k].mean(),
            "std_logit_diff": conditions[k].std(),
        }
        for k in labels
    ]
    pd.DataFrame(rows).to_csv(
        "results/s2_inhibition/si_combined_effects.csv", index=False
    )
    print(
        "Saved plots/s2_inhibition/fig4c.png and"
        " results/s2_inhibition/si_combined_effects.csv"
    )


if __name__ == "__main__":
    run()
