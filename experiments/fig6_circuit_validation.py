# indirect-object-identification/experiments/fig6_circuit_validation.py
"""Figure 6: Circuit faithfulness and completeness via mean ablation.

Uses IOIDataset("mixed", N=1000) so ABC means are averaged over template
groups (examples sharing the same syntactic template) rather than per-example
singletons, matching the method described in the paper.
"""

import os
import sys
import json
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from utils import load_model
from metrics import logit_diff
from circuit import CIRCUIT, SEQ_POS_TO_KEEP, compute_means, run_with_mean_ablation
from ioi_dataset import IOIDataset


def run():
    torch.set_grad_enabled(False)
    model = load_model()

    ioi = IOIDataset("mixed", N=1000, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))

    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()

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

    with model.trace({"input_ids": ioi.toks.long()}):
        full_logits = model.lm_head.output.save()
    full_ld = ld(full_logits.cpu())

    faithful_logits = run_with_mean_ablation(
        model, ioi.toks.long(), means, CIRCUIT, SEQ_POS_TO_KEEP, ioi.word_idx
    )
    faithful_ld = ld(faithful_logits.cpu())

    complete_logits = run_with_mean_ablation(
        model,
        ioi.toks.long(),
        means,
        {k: [] for k in CIRCUIT},
        SEQ_POS_TO_KEEP,
        ioi.word_idx,
    )
    complete_ld = ld(complete_logits.cpu())

    faith_ratio = faithful_ld / (full_ld + 1e-8)
    compl_ratio = complete_ld / (full_ld + 1e-8)

    print("\n[Figure 6]")
    print(f"  Full:              {full_ld:.4f}  (paper 3.56)")
    print(f"  Circuit only:      {faithful_ld:.4f}  (paper 3.10)")
    print(f"  Circuit ablated:   {complete_ld:.4f}  (expected ~0)")
    print(f"  Faithfulness:      {faith_ratio:.1%}  (paper 87%)")
    print(f"  Completeness rem.: {compl_ratio:.1%}  (target <5%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Full", "Circuit only\n(faithful)", "Circuit ablated\n(complete)"],
        [full_ld, faithful_ld, complete_ld],
        color=["#4C72B0", "#55A868", "#C44E52"],
    )
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Logit diff (IO − S)")
    ax.set_title("Circuit Faithfulness & Completeness (Figure 6)")
    for b, v in zip(bars, [full_ld, faithful_ld, complete_ld]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center")
    plt.tight_layout()
    os.makedirs("plots/circuit", exist_ok=True)
    plt.savefig("plots/circuit/fig6.png", dpi=150)
    plt.close()

    os.makedirs("results/circuit", exist_ok=True)
    with open("results/circuit/faithfulness.json", "w") as f:
        json.dump(
            {
                "full": full_ld,
                "faithful": faithful_ld,
                "complete": complete_ld,
                "faithfulness": faith_ratio,
                "completeness_remaining": compl_ratio,
            },
            f,
            indent=2,
        )

    assert faith_ratio > 0.85, f"Faithfulness {faith_ratio:.1%} < 85% (paper 87%)"
    assert abs(compl_ratio) < 0.10, (
        f"Completeness remainder {compl_ratio:.1%} outside ±10%"
    )
    print("PASS")


if __name__ == "__main__":
    run()
