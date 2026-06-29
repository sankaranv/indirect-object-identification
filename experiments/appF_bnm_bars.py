"""Appendix F / Fig 15: BNM bars — head→logit causal effect before and after NM knockout."""

import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from utils import load_model, clear_cache
from metrics import logit_diff
from ioi_dataset import IOIDataset
from circuit import compute_means

NM_HEADS = [(9, 9), (10, 0), (9, 6)]
NNM_HEADS = [(10, 7), (11, 10)]
BNM_HEADS = [(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)]
SI_HEADS = [(7, 3), (7, 9), (8, 6), (8, 10)]

HEAD_COLOR: dict = {}
for _lh in NM_HEADS:
    HEAD_COLOR[_lh] = "#2196F3"  # blue
for _lh in NNM_HEADS:
    HEAD_COLOR[_lh] = "#F44336"  # red
for _lh in BNM_HEADS:
    HEAD_COLOR[_lh] = "#64B5F6"  # light blue
for _lh in SI_HEADS:
    HEAD_COLOR[_lh] = "#FF9800"  # orange
_OTHER_COLOR = "#BDBDBD"

TOP_N = 20


def _patching_loop(
    model,
    clean_toks: torch.Tensor,
    corr_toks: torch.Tensor,
    metric_fn,
    nm_means: dict | None = None,
) -> dict:
    """Head-to-logit causal patching loop — no patching.py modification needed.

    For each (layer, head): run model with that head's z-slice patched from
    corr_toks; all other layers frozen at clean z.  If nm_means is provided
    (dict {(l,h): Tensor[N, seq, d_head]}), additionally mean-ablate those
    heads.  The ablations are applied within the ascending-layer loop to satisfy
    nnsight's ordering constraint (no out-of-order proxy writes).

    For non-sender NM layers: build a combined tensor (clone of clean_z with NM
    slices replaced) and write it once — avoids the need for a second write to
    the same proxy.

    For the sender layer: write sender slice first (from corrupted), then write
    any NM-head slices at the same layer (different, non-overlapping slices).

    Returns: {(l, h): float} of mean causal effects (patched − clean).
    """
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    # ── Cache corrupted z ────────────────────────────────────────────────────
    corr_z: dict = {}
    with model.trace({"input_ids": corr_toks}):
        for layer in range(n_layers):
            corr_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

    # Match nm_means device to saved activations so slice-writes don't fail.
    act_device = corr_z[0].device
    if nm_means is not None:
        nm_means = {k: v.to(act_device) for k, v in nm_means.items()}

    # ── Cache clean z and baseline metric ───────────────────────────────────
    clean_z: dict = {}
    with model.trace({"input_ids": clean_toks}):
        for layer in range(n_layers):
            clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
        clean_logits = model.lm_head.output.save()
    clean_m = metric_fn(clean_logits.cpu())
    del clean_logits
    clear_cache()

    # Pre-group NM heads by layer for O(1) lookup inside the per-sender loop.
    nm_by_layer: dict = {}
    if nm_means is not None:
        for nm_l, nm_h in NM_HEADS:
            nm_by_layer.setdefault(nm_l, []).append(nm_h)

    scores: dict = {}
    for sl in range(n_layers):
        for sh in range(n_heads):
            z_sl = slice(sh * d_head, (sh + 1) * d_head)

            with model.trace({"input_ids": clean_toks}):
                # Process layers in ascending order (nnsight ordering requirement).
                for layer in range(n_layers):
                    if layer == sl:
                        # Patch sender head's slice from corrupted.
                        model.transformer.h[layer].attn.c_proj.input[..., z_sl] = (
                            corr_z[sl][..., z_sl]
                        )
                        # For NM heads at the sender layer (different head), write their
                        # mean-ablated slice.  These slices don't overlap with z_sl (sh != nm_h).
                        if nm_by_layer.get(layer):
                            for nm_h in nm_by_layer[layer]:
                                if nm_h != sh:
                                    nm_sl = slice(nm_h * d_head, (nm_h + 1) * d_head)
                                    model.transformer.h[layer].attn.c_proj.input[
                                        ..., nm_sl
                                    ] = nm_means[(layer, nm_h)]
                    else:
                        if nm_by_layer.get(layer):
                            # Build combined tensor in Python: clone clean_z, override NM slices.
                            # Single write — avoids a second proxy-write that could be out-of-order.
                            z_comb = clean_z[layer].clone()
                            for nm_h in nm_by_layer[layer]:
                                nm_sl = slice(nm_h * d_head, (nm_h + 1) * d_head)
                                z_comb[..., nm_sl] = nm_means[(layer, nm_h)]
                            model.transformer.h[layer].attn.c_proj.input[...] = z_comb
                        else:
                            model.transformer.h[layer].attn.c_proj.input[...] = clean_z[
                                layer
                            ]

                patched_logits = model.lm_head.output.save()

            patched_m = metric_fn(patched_logits.cpu())
            scores[(sl, sh)] = (patched_m - clean_m).mean().item()
            del patched_logits
            clear_cache()

    del corr_z, clean_z
    clear_cache()
    return scores


def _bar_chart(scores: dict, title: str, fname: str) -> None:
    """Plot top-N heads by |effect|, colored by head type."""
    items = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N]
    labels = [f"{layer}.{head}" for (layer, head), _ in items]
    effects = [v for _, v in items]
    colors = [HEAD_COLOR.get((layer, head), _OTHER_COLOR) for (layer, head), _ in items]

    fig, ax = plt.subplots(figsize=(max(10, TOP_N * 0.6), 5))
    ax.bar(range(len(items)), effects, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_ylabel("Causal effect (patched − clean logit diff)")
    ax.set_title(title)

    legend_entries = [
        mpatches.Patch(color="#2196F3", label="Name Mover"),
        mpatches.Patch(color="#F44336", label="Negative NM"),
        mpatches.Patch(color="#64B5F6", label="Backup NM"),
        mpatches.Patch(color="#FF9800", label="S2 Inhibition"),
        mpatches.Patch(color=_OTHER_COLOR, label="Other"),
    ]
    ax.legend(handles=legend_entries, loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")


def run() -> None:
    torch.set_grad_enabled(False)
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))

    N = len(ioi)
    end_pos = ioi.word_idx["end"].long()

    def metric_fn(logits: torch.Tensor) -> torch.Tensor:
        """logits: [N, seq, vocab] → per-example logit diff [N]."""
        end_logits = logits[torch.arange(N), end_pos]  # [N, vocab]
        return logit_diff(end_logits, ioi.io_tokenIDs, ioi.s_tokenIDs)

    # ── Before NM knockout ───────────────────────────────────────────────────
    scores_before = _patching_loop(
        model, ioi.toks.long(), abc.toks.long(), metric_fn, nm_means=None
    )
    _bar_chart(
        scores_before,
        "Head → logit causal effect, before NM knockout",
        "plots/backup/fig15_before.png",
    )

    # ── Compute NM head means from ABC dataset ───────────────────────────────
    full_means = compute_means(model, abc.toks.long(), abc.groups)
    # full_means: [n_layers, N, seq, n_heads, d_head]  (stored on CPU by compute_means)

    nm_means: dict = {}
    for nm_l, nm_h in NM_HEADS:
        # Shape [N, seq, d_head] — template-group mean z for this NM head.
        nm_means[(nm_l, nm_h)] = full_means[nm_l, :, :, nm_h, :].contiguous()
    del full_means
    clear_cache()

    # ── After NM knockout ────────────────────────────────────────────────────
    scores_after = _patching_loop(
        model, ioi.toks.long(), abc.toks.long(), metric_fn, nm_means=nm_means
    )
    _bar_chart(
        scores_after,
        "Head → logit causal effect, after NM knockout",
        "plots/backup/fig15_after.png",
    )


if __name__ == "__main__":
    run()
