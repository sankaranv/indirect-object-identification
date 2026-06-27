"""NNM: negative-effect heads from Phase 1 CSV.
Backup NMs: heads whose contribution to logit diff rises when primary NMs are ablated.
NNM expected: (10,7), (11,10)
BNM expected: (10,10),(10,6),(10,2),(10,1),(11,2),(9,7),(9,0),(11,9)
"""
import os, sys, json, random, torch, einops
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from utils import load_model
from ioi_dataset import IOIDataset

NM_HEADS     = [(9, 9), (10, 0), (9, 6)]
EXPECTED_NNM = {(10, 7), (11, 10)}
EXPECTED_BNM = {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)}
NM_CSV       = "results/name_movers/head_to_logits_causal_effect.csv"


def _per_head_logit_diff_contribution(
    model, ioi: IOIDataset, ablate=None
) -> torch.Tensor:
    """[n_layers, n_heads] — each head's contribution to IO-S logit diff at END.

    Computes head_output @ W_U projected onto the IO-S direction, averaged over examples.
    ablate: optional list of (layer, head) pairs to zero out before their z contributes to c_proj.

    nnsight does not allow saving and writing to c_proj.input at the same layer in the
    same trace.  We therefore use a multi-pass strategy:

      Pass 1 (clean):  save z for all layers -> z_clean[0..n_layers-1].

      For each ablation layer al (sorted):
        - z[al] is computed from the residual *entering* layer al, which is unaffected by
          the ablation at al itself.  So z[al]_ablated = z_clean[al] (if al is the first
          ablation layer) or z from a prior-ablation trace (if earlier layers were also
          ablated), with ablated head slices then zeroed in Python.
        - A dedicated trace (write = all prior ablations; save = al) avoids the conflict.
        - Layers between consecutive ablation checkpoints are saved in the same trace used
          for the earlier checkpoint (writes and saves target different layers).

    This gives the correct z at every layer under the cascading ablation.
    """
    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    N        = len(ioi)
    end_pos  = ioi.word_idx["end"]

    W_U    = model.lm_head.weight.detach()                          # [vocab, d_model]
    io_dir = W_U[ioi.io_tokenIDs] - W_U[ioi.s_tokenIDs]            # [N, d_model]

    if ablate is None:
        # Single clean pass: one trace, no writes.
        z_list = [None] * n_layers
        with model.trace({"input_ids": ioi.toks.long()}):
            for layer in range(n_layers):
                z_list[layer] = model.transformer.h[layer].attn.c_proj.input.save()
    else:
        # Multi-pass ablated: never save and write at the same layer in one trace.
        abl_by_layer: dict = {}
        for al, ah in ablate:
            abl_by_layer.setdefault(al, []).append(ah)
        abl_layers = sorted(abl_by_layer)

        # Pass 1: clean trace -- baseline z for all layers.
        z_clean = [None] * n_layers
        with model.trace({"input_ids": ioi.toks.long()}):
            for layer in range(n_layers):
                z_clean[layer] = model.transformer.h[layer].attn.c_proj.input.save()

        # z_list starts as clean; layers 0..(a0-1) need no update (no upstream ablations).
        z_list = list(z_clean)
        # written_z: {layer -> modified tensor} injected into downstream traces.
        written_z: dict = {}

        for idx, al in enumerate(abl_layers):
            # Get z at ablation layer al from the correct (prior-ablated) residual.
            if idx == 0:
                # First ablation layer: no prior ablations -> residual is clean.
                z_al_pre = z_clean[al]
            else:
                # Prior ablations modified the residual; run a dedicated trace.
                # Writes: all previously-applied ablations (layers != al).
                # Save:   layer al -- NO CONFLICT (different layers).
                with model.trace({"input_ids": ioi.toks.long()}):
                    for wl, wz in written_z.items():
                        model.transformer.h[wl].attn.c_proj.input[...] = wz
                    _z_save = model.transformer.h[al].attn.c_proj.input.save()
                z_al_pre = _z_save

            # Apply this layer's ablation: zero out the nominated head slices.
            z_al_mod = z_al_pre.clone()
            for ah in abl_by_layer[al]:
                z_al_mod[..., ah * d_head : (ah + 1) * d_head] = 0.0
            z_list[al]    = z_al_mod
            written_z[al] = z_al_mod   # register for downstream injection

            # Save z for layers between this ablation and the next checkpoint.
            # Writes: all ablations so far (layers <= al).
            # Saves:  layers al+1..next_al-1 (or al+1..n_layers-1 if last checkpoint).
            # Write layers < save layers -> NO CONFLICT.
            if idx + 1 < len(abl_layers):
                save_range = range(al + 1, abl_layers[idx + 1])
            else:
                save_range = range(al + 1, n_layers)

            if save_range:
                _z_saved: dict = {}
                with model.trace({"input_ids": ioi.toks.long()}):
                    for wl, wz in written_z.items():
                        model.transformer.h[wl].attn.c_proj.input[...] = wz
                    for layer in save_range:
                        _z_saved[layer] = model.transformer.h[layer].attn.c_proj.input.save()
                for layer in save_range:
                    z_list[layer] = _z_saved[layer]

    # Compute per-head contributions.
    out = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        z = z_list[layer]
        W_O   = model.transformer.h[layer].attn.c_proj.weight.detach()  # [d_model, d_model]
        W_O_h = W_O.view(n_heads, d_head, model.config.n_embd)
        z_end = z[torch.arange(N), end_pos].view(N, n_heads, d_head)
        head_out = einops.einsum(z_end, W_O_h, "n nh dh, nh dh dm -> n nh dm")
        contribution = einops.einsum(head_out, io_dir, "n nh dm, n dm -> nh") / N
        out[layer] = contribution.cpu()
    return out


def run(nnm_threshold=-0.05, bnm_threshold=0.1):
    model = load_model()
    random.seed(1)
    np.random.seed(1)
    ioi   = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)

    df  = pd.read_csv(NM_CSV)
    nnm = {(int(r['layer']), int(r['head'])) for _, r in df.iterrows() if r['causal_effect'] < nnm_threshold}
    print(f"NNM: {sorted(nnm)}  expected {sorted(EXPECTED_NNM)}")
    print("PASS" if EXPECTED_NNM <= nnm else f"WARNING missing {EXPECTED_NNM - nnm}")

    print("\nComputing per-head logit diff contributions (original)...")
    contrib_orig = _per_head_logit_diff_contribution(model, ioi)
    print("Computing per-head logit diff contributions (NMs ablated)...")
    contrib_abl  = _per_head_logit_diff_contribution(model, ioi, ablate=NM_HEADS)
    contrib_diff = contrib_abl - contrib_orig

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, data, title in zip(
        axes, [contrib_orig, contrib_abl, contrib_diff],
        ["Contribution", "Contribution (NMs ablated)", "Change in contribution"],
    ):
        vm = max(data.abs().max().item(), 1e-6)
        im = ax.imshow(data.numpy(), aspect="auto", cmap="RdBu", vmin=-vm, vmax=vm)
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs("plots/backup", exist_ok=True)
    plt.savefig("plots/backup/neg_backup.png", dpi=150); plt.close()

    n_layers, n_heads = contrib_diff.shape
    nm_set = set(NM_HEADS)
    bnm = {(l, h) for l in range(n_layers) for h in range(n_heads)
           if (l, h) not in nm_set and contrib_diff[l, h].item() > bnm_threshold}
    print(f"\nBNM: {sorted(bnm)}  expected {sorted(EXPECTED_BNM)}")
    missing = EXPECTED_BNM - bnm
    print("PASS" if not missing else f"WARNING missing {missing}")

    os.makedirs("results/backup", exist_ok=True)
    with open("results/backup/all_nm_types.json", "w") as f:
        json.dump({"name_mover": [list(h) for h in NM_HEADS],
                   "negative":   sorted([list(h) for h in nnm]),
                   "backup":     sorted([list(h) for h in bnm])}, f, indent=2)
    return nnm, bnm


if __name__ == "__main__":
    run()
