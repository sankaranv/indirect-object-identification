"""Appendix A: Disentangle token vs positional signal in S-Inhibition head outputs.

Six counterfactual datasets (3 token conditions x 2 position conditions):
  Token: original S name | random name | full IO<->S swap
  Position: original | inverted (IO and S1 word positions swapped via ("IO","S1"))

For each dataset the SI head z-vectors are activation-patched into the clean IOI
run and the resulting logit diff is recorded (Fig 9 heatmap).

Fig 10 shows how the Name Mover heads' attention to IO/S1/S2 changes before vs.
after patching SI heads from two specific counterfactual datasets.

Run from indirect-object-identification/:
    python experiments/appA_signal_decomposition.py
"""

import copy
import os
import random
import sys

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from ioi_dataset import IOIDataset
from metrics import logit_diff
from utils import clear_cache, load_model

# S-Inhibition heads (Appendix A of the IOI paper)
SI_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]
# Name-Mover heads
NM_HEADS = [(9, 9), (10, 0), (9, 6)]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _gen_name_swap_dataset(ioi: IOIDataset) -> IOIDataset:
    """Return an IOIDataset where IO and S names are fully swapped in the text.

    All occurrences of the IO name become the S name and vice versa. The word
    position indices (word_idx) are frozen to the original IOI layout so that
    logit_diff can still be computed with the original token IDs.
    """
    new_prompts = []
    for p in ioi.ioi_prompts:
        p2 = copy.deepcopy(p)
        io, s = p["IO"], p["S"]
        # Use a sentinel to avoid double-replacement
        text = (
            p["text"]
            .replace(io, "\x00IOMARK\x00")
            .replace(s, io)
            .replace("\x00IOMARK\x00", s)
        )
        p2["text"] = text
        p2["IO"] = s
        p2["S"] = io
        new_prompts.append(p2)
    return IOIDataset(
        prompt_type=ioi.prompt_type,
        N=ioi.N,
        tokenizer=ioi.tokenizer,
        prompts=new_prompts,
        prepend_bos=ioi.prepend_bos,
        manual_word_idx=ioi.word_idx,  # preserve original positional bookkeeping
    )


def build_counterfactual_datasets(ioi: IOIDataset) -> dict:
    """Return {(token_signal, pos_signal): IOIDataset} for all 6 conditions.

    Token signals:
      "original" — S name unchanged
      "random"   — S name replaced by a random name (via ("S","RAND"))
      "swapped"  — IO and S names fully exchanged everywhere in the text

    Position signals:
      "original" — IO and S1 positions unchanged
      "inverted" — IO and S1 word positions swapped (via ("IO","S1"))

    Note: gen_flipped_prompts(("S","IO")) is not implemented in the underlying
    standalone function, so we implement the name swap manually in
    _gen_name_swap_dataset, which uses manual_word_idx to keep word positions
    consistent for logit_diff evaluation.
    """
    # -- Token variants (original positions) --
    orig_tok = ioi
    rand_tok = ioi.gen_flipped_prompts(("S", "RAND"))
    swap_tok = _gen_name_swap_dataset(ioi)

    # -- Position inversion: swap IO and S1 word positions --
    inv_pos_ioi = ioi.gen_flipped_prompts(("IO", "S1"))
    inv_pos_rand = inv_pos_ioi.gen_flipped_prompts(("S", "RAND"))
    inv_pos_swap = _gen_name_swap_dataset(inv_pos_ioi)

    return {
        ("original", "original"): orig_tok,
        ("random",   "original"): rand_tok,
        ("swapped",  "original"): swap_tok,
        ("original", "inverted"): inv_pos_ioi,
        ("random",   "inverted"): inv_pos_rand,
        ("swapped",  "inverted"): inv_pos_swap,
    }


# ---------------------------------------------------------------------------
# Activation patching helpers
# ---------------------------------------------------------------------------


def _build_patch_layers() -> dict:
    patch_layers = {}
    for l, h in SI_HEADS:
        patch_layers.setdefault(l, []).append(h)
    return patch_layers


def _patch_si_heads_logits(
    model, clean_toks: torch.Tensor, src_toks: torch.Tensor,
    patch_layers: dict, d_head: int,
) -> torch.Tensor:
    """Patch SI head z-vectors from src_toks into a clean forward pass.

    Returns logits [N, seq, vocab] as a CPU tensor.
    """
    # Cache z (c_proj input) from the source/counterfactual run
    src_z = {}
    with model.trace({"input_ids": src_toks}):
        for l in patch_layers:
            src_z[l] = model.transformer.h[l].attn.c_proj.input.save()

    # Run clean tokens, replacing SI-head slices with source z, save logits
    with model.trace({"input_ids": clean_toks}):
        for l, heads in patch_layers.items():
            for h in heads:
                sl = slice(h * d_head, (h + 1) * d_head)
                model.transformer.h[l].attn.c_proj.input[..., sl] = src_z[l][..., sl]
        logits = model.lm_head.output.save()

    return logits.cpu()


# ---------------------------------------------------------------------------
# Figure 10 helpers
# ---------------------------------------------------------------------------


def _nm_attention_before_after(
    model, ioi: IOIDataset, patch_ds: IOIDataset,
    patch_layers: dict, d_head: int,
) -> tuple:
    """Compute mean NM-head attention to IO/S1/S2 before and after SI patching.

    Returns (before_dict, after_dict) where each maps "IO"/"S1"/"S2" -> float.
    """
    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()
    io_pos  = ioi.word_idx["IO"].long()
    s1_pos  = ioi.word_idx["S"].long()
    s2_pos  = ioi.word_idx["S2"].long()

    # Cache SI activations from the counterfactual dataset
    src_z = {}
    with model.trace({"input_ids": patch_ds.toks.long()}):
        for l in patch_layers:
            src_z[l] = model.transformer.h[l].attn.c_proj.input.save()
    clear_cache()

    # Before: clean IOI run, no patches — get NM attention weights
    with model.trace({"input_ids": ioi.toks.long()}, output_attentions=True):
        w9_bef  = model.transformer.h[9].attn.output[1].save()
        w10_bef = model.transformer.h[10].attn.output[1].save()

    # After: clean IOI with SI patches applied — get NM attention weights
    with model.trace({"input_ids": ioi.toks.long()}, output_attentions=True):
        for l, heads in patch_layers.items():
            for h in heads:
                sl = slice(h * d_head, (h + 1) * d_head)
                model.transformer.h[l].attn.c_proj.input[..., sl] = src_z[l][..., sl]
        w9_aft  = model.transformer.h[9].attn.output[1].save()
        w10_aft = model.transformer.h[10].attn.output[1].save()

    # Move to CPU for indexing
    attn_bef = {9: w9_bef.cpu(),  10: w10_bef.cpu()}
    attn_aft = {9: w9_aft.cpu(), 10: w10_aft.cpu()}
    idx = torch.arange(N)
    end_pos = end_pos.cpu()
    io_pos  = io_pos.cpu()
    s1_pos  = s1_pos.cpu()
    s2_pos  = s2_pos.cpu()

    def _mean(attn_dict, kpos):
        """Average attention from end_pos to kpos across all NM heads."""
        total = 0.0
        for l, h in NM_HEADS:
            total += attn_dict[l][idx, h, end_pos, kpos].mean().item()
        return total / len(NM_HEADS)

    before = {
        "IO": _mean(attn_bef, io_pos),
        "S1": _mean(attn_bef, s1_pos),
        "S2": _mean(attn_bef, s2_pos),
    }
    after = {
        "IO": _mean(attn_aft, io_pos),
        "S1": _mean(attn_aft, s1_pos),
        "S2": _mean(attn_aft, s2_pos),
    }
    return before, after


def _plot_fig10(
    model, ioi: IOIDataset, patch_ds: IOIDataset,
    out_path: str, title: str,
    patch_layers: dict, d_head: int,
) -> None:
    before, after = _nm_attention_before_after(
        model, ioi, patch_ds, patch_layers, d_head
    )

    labels = ["IO", "S1", "S2"]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(x - 0.2, [before[k] for k in labels], 0.35,
           label="Before patching", color="#4C72B0")
    ax.bar(x + 0.2, [after[k]  for k in labels], 0.35,
           label="After patching",  color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean attention probability")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    N       = len(ioi)
    end_pos = ioi.word_idx["end"].long()
    n_heads = model.config.n_head
    d_head  = model.config.n_embd // n_heads

    patch_layers = _build_patch_layers()

    print("Building counterfactual datasets...")
    datasets = build_counterfactual_datasets(ioi)

    # -----------------------------------------------------------------------
    # Figure 9: 3×2 heatmap of logit diff after patching SI heads
    # -----------------------------------------------------------------------
    print("Computing logit diffs (Fig 9)...")
    results_9 = {}
    for (tok_sig, pos_sig), ds in datasets.items():
        logits = _patch_si_heads_logits(
            model, ioi.toks.long(), ds.toks.long(), patch_layers, d_head
        )
        ld = logit_diff(
            logits[torch.arange(N), end_pos],
            ioi.io_tokenIDs,
            ioi.s_tokenIDs,
        ).mean().item()
        results_9[(tok_sig, pos_sig)] = ld
        print(f"  ({tok_sig:10s}, {pos_sig:10s}):  logit_diff = {ld:+.3f}")
        clear_cache()

    tok_signals = ["original", "random", "swapped"]
    pos_signals = ["original", "inverted"]
    arr = np.array([[results_9[(t, p)] for p in pos_signals] for t in tok_signals])

    # Save CSV
    os.makedirs("results/signal", exist_ok=True)
    rows = [
        {
            "token_signal": t,
            "position_original": results_9[(t, "original")],
            "position_inverted": results_9[(t, "inverted")],
        }
        for t in tok_signals
    ]
    pd.DataFrame(rows).to_csv("results/signal/fig9.csv", index=False)
    print("Saved results/signal/fig9.csv")

    # Heatmap
    os.makedirs("plots/signal", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(arr, cmap="RdBu", vmin=-4, vmax=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Original position", "Inverted position"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["Orig. S token", "Random S token", "IO↔S swap"])
    plt.colorbar(im, ax=ax, label="Logit diff after patching SI heads")
    for i in range(3):
        for j in range(2):
            val = arr[i, j]
            color = "white" if abs(val) > 2.5 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title("Logit diff: SI head signal decomposition")
    plt.tight_layout()
    plt.savefig("plots/signal/fig9.png", dpi=150)
    plt.close()
    print("Saved plots/signal/fig9.png")

    # -----------------------------------------------------------------------
    # Figure 10: NM attention before vs after patching from two datasets
    # -----------------------------------------------------------------------
    print("Computing NM attention (Fig 10, random-name dataset)...")
    _plot_fig10(
        model, ioi, datasets[("random", "original")],
        "plots/signal/fig10_random_name.png",
        "NM attention: SI patched from random-name dataset",
        patch_layers, d_head,
    )

    print("Computing NM attention (Fig 10, IO↔S swap + inverted position)...")
    _plot_fig10(
        model, ioi, datasets[("swapped", "inverted")],
        "plots/signal/fig10_io_s2_swap.png",
        "NM attention: SI patched from IO↔S + inverted-position dataset",
        patch_layers, d_head,
    )


if __name__ == "__main__":
    run()
