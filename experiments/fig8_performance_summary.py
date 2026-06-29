"""Figure 8: GPT-2 performance on IOI, ABC, and adversarial datasets."""

import os
import sys
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from utils import load_model
from metrics import logit_diff
from ioi_dataset import IOIDataset

ADV_TEMPLATES = [
    "{IO} had a good day.",
    "{IO} was enjoying the situation.",
    "{IO} was tired.",
    "{IO} enjoyed being with a friend.",
    "{IO} was an enthusiast person.",
]


def build_adversarial_dataset(ioi: IOIDataset, model):
    """Prepend a duplicate-IO clause to each IOI prompt.

    Returns:
        toks    (LongTensor [N, seq_len]): padded token ids
        end_pos (LongTensor [N]):          per-example position of the "to"
                                           token (second-to-last real token,
                                           where we read logits to predict IO)
    """
    texts = []
    for i, prompt in enumerate(ioi.ioi_prompts):
        io_name = prompt["IO"]
        tmpl = ADV_TEMPLATES[i % len(ADV_TEMPLATES)]
        prefix = tmpl.format(IO=io_name)
        texts.append(prefix + " " + prompt["text"])
    enc = model.tokenizer(texts, return_tensors="pt", padding=True)
    toks = enc.input_ids
    # IO name is the last real token; "to" is second-to-last → index -2 from end
    end_pos = enc.attention_mask.sum(-1) - 2
    return toks, end_pos


def _eval(model, toks, end_pos, io_ids, s_ids):
    """Forward-pass model on toks and return logit-diff metrics.

    Args:
        model:   nnsight LanguageModel
        toks:    LongTensor [N, seq]
        end_pos: LongTensor [N] or scalar — position where we read logits
        io_ids:  list[int] length N — IO token ids
        s_ids:   list[int] length N — S token ids

    Returns:
        (mean_logit_diff, mean_io_prob, frac_s_over_io)
    """
    N = toks.size(0)
    with model.trace({"input_ids": toks}):
        logits = model.lm_head.output.save()
    logits = logits.cpu()

    # select per-example end positions
    rows = torch.arange(N)
    ep = (
        end_pos
        if isinstance(end_pos, torch.Tensor)
        else torch.full((N,), end_pos, dtype=torch.long)
    )
    logits_at_end = logits[rows, ep]  # [N, vocab]

    ld = logit_diff(logits_at_end, io_ids, s_ids)  # [N]
    io_ids_t = torch.tensor(io_ids)
    io_probs = logits_at_end.softmax(-1)[rows, io_ids_t]  # [N]
    s_over_io_rate = (ld < 0).float().mean().item()

    return ld.mean().item(), io_probs.mean().item(), s_over_io_rate


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))

    end_pos = ioi.word_idx["end"].long()  # per-example [N]

    ioi_ld, ioi_io_prob, ioi_s_rate = _eval(
        model, ioi.toks.long(), end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs
    )

    abc_ld, abc_io_prob, abc_s_rate = _eval(
        model, abc.toks.long(), end_pos, abc.io_tokenIDs, abc.s_tokenIDs
    )

    adv_toks, adv_end_pos = build_adversarial_dataset(ioi, model)

    adv_ld, adv_io_prob, adv_s_rate = _eval(
        model, adv_toks, adv_end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs
    )

    rows = [
        {
            "dataset": "IOI (clean)",
            "logit_diff": ioi_ld,
            "io_prob": ioi_io_prob,
            "s_over_io_rate": ioi_s_rate,
        },
        {
            "dataset": "ABC (corrupted)",
            "logit_diff": abc_ld,
            "io_prob": abc_io_prob,
            "s_over_io_rate": abc_s_rate,
        },
        {
            "dataset": "Adversarial",
            "logit_diff": adv_ld,
            "io_prob": adv_io_prob,
            "s_over_io_rate": adv_s_rate,
        },
    ]
    df = pd.DataFrame(rows)

    os.makedirs("results/adversarial", exist_ok=True)
    df.to_csv("results/adversarial/performance_summary.csv", index=False)
    print("\nResults:")
    for _, r in df.iterrows():
        print(
            f"  {r['dataset']:20s}: LD={r['logit_diff']:+.3f}  IO_prob={r['io_prob']:.3f}  S_rate={r['s_over_io_rate']:.3f}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = df["dataset"].tolist()
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    metrics = [
        ("logit_diff", "Logit diff (IO − S)"),
        ("io_prob", "IO token probability"),
        ("s_over_io_rate", "Rate S predicted over IO"),
    ]
    for ax, (col, ylabel) in zip(axes, metrics):
        ax.bar(range(3), df[col].values, color=colors)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    plt.suptitle("GPT-2 Small performance across IOI dataset variants", y=1.01)
    plt.tight_layout()
    os.makedirs("plots/adversarial", exist_ok=True)
    plt.savefig("plots/adversarial/fig8.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved plots/adversarial/fig8.png")


if __name__ == "__main__":
    run()
