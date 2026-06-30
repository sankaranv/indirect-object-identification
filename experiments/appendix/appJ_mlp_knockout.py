"""Appendix J / Fig 19: Direct and indirect MLP effects on logit diff."""

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
from model import clear_cache, load_model


def _bar_chart(effects: list, title: str, fname: str) -> None:
    n = len(effects)
    colors = ["#E57373" if v < 0 else "#64B5F6" for v in effects]
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    ax.bar(range(n), effects, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"MLP{layer}" for layer in range(n)], rotation=45, fontsize=9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Change in logit diff")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved {fname}")


def run() -> tuple:
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
    io_ids = ioi.io_tokenIDs
    s_ids = ioi.s_tokenIDs
    n_layers = len(model.transformer.h)

    def metric(logits: torch.Tensor) -> torch.Tensor:
        """logits: [N, seq, vocab] → per-example logit diff [N]."""
        end_logits = logits[torch.arange(N), end_pos]  # [N, vocab]
        return logit_diff(end_logits, io_ids, s_ids)  # [N]

    # ── Baseline ─────────────────────────────────────────────────────────────
    with model.trace({"input_ids": ioi.toks.long()}):
        base_logits = model.lm_head.output.save()
    base_ld = metric(base_logits.cpu()).mean().item()
    del base_logits
    clear_cache()
    print(f"Baseline logit diff: {base_ld:.4f}")

    # ── Total effect: patch MLP_l output from ABC, everything else clean ─────
    # Indirect effect is then: total − direct (per Wang et al. 2022 Appendix J).
    total_effects = []
    for target_l in range(n_layers):
        with model.trace({"input_ids": abc.toks.long()}):
            abc_mlp_total = model.transformer.h[target_l].mlp.output.save()
        with model.trace({"input_ids": ioi.toks.long()}):
            model.transformer.h[target_l].mlp.output[...] = abc_mlp_total
            logits = model.lm_head.output.save()
        ld = metric(logits.cpu()).mean().item()
        total_effects.append(ld - base_ld)
        del abc_mlp_total, logits
        clear_cache()

    # ── Direct effect: path-patch MLP_l from ABC, freeze attn z + other MLPs ─
    # Cache clean attn z (separate trace from MLP saves to avoid nnsight conflicts).
    clean_z: dict = {}
    with model.trace({"input_ids": ioi.toks.long()}):
        for layer in range(n_layers):
            clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

    # Cache clean MLP outputs in a separate trace.
    clean_mlp: dict = {}
    with model.trace({"input_ids": ioi.toks.long()}):
        for layer in range(n_layers):
            clean_mlp[layer] = model.transformer.h[layer].mlp.output.save()

    direct_effects = []
    for target_l in range(n_layers):
        # Cache target-layer MLP output on ABC.
        with model.trace({"input_ids": abc.toks.long()}):
            abc_mlp = model.transformer.h[target_l].mlp.output.save()

        # Patch: freeze all attn head z's at clean values; swap target MLP from ABC;
        # keep all other MLP outputs at clean values → isolates direct path
        # MLP_l → logit.
        with model.trace({"input_ids": ioi.toks.long()}):
            for layer in range(n_layers):
                model.transformer.h[layer].attn.c_proj.input[...] = clean_z[layer]
                if layer == target_l:
                    model.transformer.h[layer].mlp.output[...] = abc_mlp
                else:
                    model.transformer.h[layer].mlp.output[...] = clean_mlp[layer]
            patched_logits = model.lm_head.output.save()

        ld = metric(patched_logits.cpu()).mean().item()
        direct_effects.append(ld - base_ld)
        del patched_logits, abc_mlp
        clear_cache()

    del clean_z, clean_mlp
    clear_cache()

    _bar_chart(
        direct_effects,
        "MLP direct effect (path patching)",
        "plots/mlp/fig19_direct.png",
    )

    # Indirect effect = total − direct (Wang et al. 2022 Appendix J definition).
    indirect_effects = [t - d for t, d in zip(total_effects, direct_effects)]
    _bar_chart(
        indirect_effects,
        "MLP indirect effect (total − direct)",
        "plots/mlp/fig19_indirect.png",
    )

    return indirect_effects, direct_effects


if __name__ == "__main__":
    run()
