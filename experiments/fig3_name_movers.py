"""Figure 3: Name Mover heads identified via head→logit path patching,
attention patterns, unembed projections, and OV copy strength.
Expected: (9,9), (10,0), (9,6)
"""

import csv
import os
import sys
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from utils import load_model
from metrics import logit_diff
from patching import path_patch_head_to_logits
from analysis import ov_copy_strength
from ioi_dataset import IOIDataset

EXPECTED = {(9, 9), (10, 0), (9, 6)}
# With paper sign convention (patched − clean), helpful heads have NEGATIVE causal effects.
THRESHOLD = -0.01


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(1)
    np.random.seed(1)
    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    # ABC baseline: replace IO with random, then S with random
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))
    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()  # [N]

    def metric(logits):
        return logit_diff(
            logits[torch.arange(N), end_pos],
            ioi.io_tokenIDs,
            ioi.s_tokenIDs,
        )

    # Head→logit causal effects via path patching
    result = path_patch_head_to_logits(
        model,
        ioi.toks.long(),
        abc.toks.long(),
        metric,
    )
    effects = result.scores

    # Save full causal-effect matrix for downstream scripts (neg_backup_nm, fig3b, etc.)
    os.makedirs("results/name_movers", exist_ok=True)
    with open(
        "results/name_movers/head_to_logits_causal_effect.csv", "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=["layer", "head", "causal_effect"])
        w.writeheader()
        for (layer, head), v in sorted(effects.items()):
            w.writerow({"layer": layer, "head": head, "causal_effect": v})

    # NMs have negative causal effects (patched − clean): corrupting them hurts logit diff.
    candidate_heads = [
        (layer, head) for (layer, head), e in effects.items() if e < THRESHOLD
    ]

    # OV copy strength for candidates
    copy = {
        (layer, head): ov_copy_strength(model, layer, head)
        for layer, head in candidate_heads
    }

    # --- plots ---
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head

    effect_arr = np.zeros((n_layers, n_heads))
    for (layer, head), e in effects.items():
        effect_arr[layer, head] = e

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    vmax = np.abs(effect_arr).max()
    im = axes[0].imshow(effect_arr, cmap="RdBu", vmin=-vmax, vmax=vmax)
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Head→logit causal effect (Figure 3a)")
    plt.colorbar(im, ax=axes[0])

    if copy:
        lh_labels = [f"{layer}.{head}" for layer, head in sorted(copy)]
        axes[1].bar(lh_labels, [copy[k] for k in sorted(copy)])
        axes[1].set_ylabel("OV copy strength")
        axes[1].set_title("OV copy strength for candidate NM heads (Figure 3c)")
        axes[1].set_xticklabels(lh_labels, rotation=45, ha="right")

    plt.tight_layout()
    os.makedirs("plots/name_movers", exist_ok=True)
    plt.savefig("plots/name_movers/fig3.png", dpi=150)
    plt.close()

    # Gate
    found = set(candidate_heads)
    missing = EXPECTED - found
    extra = found - EXPECTED
    print(f"Found:    {sorted(found)}")
    print(f"Expected: {sorted(EXPECTED)}")
    if extra:
        print(f"WARNING: extra heads (not in EXPECTED): {sorted(extra)}")
    print("WARNING: missing " + str(missing) if missing else "PASS")

    os.makedirs("results/name_movers", exist_ok=True)
    with open("results/name_movers/identified_heads.json", "w") as f:
        json.dump(sorted([list(h) for h in found]), f)

    return found


if __name__ == "__main__":
    run()
