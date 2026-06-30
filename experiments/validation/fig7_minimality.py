# indirect-object-identification/experiments/fig7_minimality.py
"""Figure 7: Minimality — every circuit head has score > 0 when companions are
ablated."""

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import N, SEED

from circuit import (
    CIRCUIT,
    K_FOR_EACH_COMPONENT,
    SEQ_POS_TO_KEEP,
    run_with_ablation,
)
from ablation import compute_means, mean_ablation
from ioi_dataset import IOIDataset
from metrics import logit_diff
from model import load_model


def ablate_heads(base_circuit, heads_to_remove):
    result = {}
    for k, heads in base_circuit.items():
        result[k] = [
            (layer, head)
            for (layer, head) in heads
            if (layer, head) not in heads_to_remove
        ]
    return result


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
    end_pos = ioi.word_idx["end"]

    def ld(logits):
        return (
            logit_diff(
                logits[torch.arange(N), end_pos],
                ioi.io_tokenIDs,
                ioi.s_tokenIDs,
            )
            .mean()
            .item()
        )

    means = compute_means(model, abc.toks.long(), abc.groups)
    ablation = mean_ablation(means)

    with model.trace({"input_ids": ioi.toks.long()}):
        full_logits = model.lm_head.output.save()
    full_ld = ld(full_logits.cpu())

    print(f"Full model LD: {full_ld:.4f}")

    scores = {}
    head_to_type = {}
    for head_type, heads in CIRCUIT.items():
        for h in heads:
            head_to_type[h] = head_type

    for head, K in K_FOR_EACH_COMPONENT.items():
        # C \ K (circuit keeping head, removing companions)
        circ_minus_K = ablate_heads(CIRCUIT, K)
        # C \ K ∪ {head} (circuit removing companions AND head)
        circ_minus_K_v = ablate_heads(CIRCUIT, K | {head})

        logits_K = run_with_ablation(
            model, ioi.toks.long(), ablation, circ_minus_K, SEQ_POS_TO_KEEP, ioi.word_idx
        )
        logits_K_v = run_with_ablation(
            model, ioi.toks.long(), ablation, circ_minus_K_v, SEQ_POS_TO_KEEP, ioi.word_idx
        )

        score = abs(ld(logits_K.cpu()) - ld(logits_K_v.cpu()))
        scores[head] = score
        status = "OK" if score > 0 else "FAIL"
        print(
            f"  {head_to_type.get(head, '?'):25s} {head}  "
            f"abs_score={score:+.4f}  {status}"
        )

    # Plot
    all_heads = list(K_FOR_EACH_COMPONENT.keys())
    all_scores = [scores[h] for h in all_heads]
    labels = [f"{head_to_type.get(h, '?')[:8]}\n{h}" for h in all_heads]

    fig, ax = plt.subplots(figsize=(max(12, len(all_heads) * 0.5), 5))
    colors = ["#55A868" if s > 0 else "#C44E52" for s in all_scores]
    ax.bar(range(len(all_heads)), all_scores, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(all_heads)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Minimality score (LD drop when head removed)")
    ax.set_title("Circuit Minimality (Figure 7)")
    plt.tight_layout()
    os.makedirs("plots/circuit", exist_ok=True)
    plt.savefig("plots/circuit/fig7.png", dpi=150)
    plt.close()

    os.makedirs("results/circuit", exist_ok=True)
    with open("results/circuit/minimality.json", "w") as f:
        json.dump({str(h): scores[h] for h in all_heads}, f, indent=2)

    failed = [h for h, s in scores.items() if s <= 0]
    print(
        f"\n[Figure 7]  {len(all_heads)} heads evaluated;"
        f" {len(failed)} failed minimality"
    )
    assert not failed, f"Minimality FAIL: {failed} have score ≤ 0"
    print("PASS")


if __name__ == "__main__":
    run()
