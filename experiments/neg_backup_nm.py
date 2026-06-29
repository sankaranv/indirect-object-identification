"""NNM: negative-effect heads from Phase 1 CSV.
Backup NMs: heads that compensate when primary NMs are ablated.
NNM expected: (10,7), (11,10)
BNM expected: (10,10),(10,6),(10,2),(10,1),(11,2),(9,7),(9,0),(11,9)

BNM identification uses path patching in the NM-ablated model (paper Sec 3.2.4):
NM heads are mean-ablated in every trace; each candidate sender's z is then
corrupted from ABC. This correctly surfaces same-layer heads (9,0) and (9,7),
which are invisible to contribution-diff because ablating NMs at layer 9 does
not change the residual entering layer 9.
"""
import os, sys, json, random, torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from utils import load_model, clear_cache
from metrics import logit_diff
from circuit import compute_means
from ioi_dataset import IOIDataset

NM_HEADS     = [(9, 9), (10, 0), (9, 6)]
EXPECTED_NNM = {(10, 7), (11, 10)}
EXPECTED_BNM = {(10, 10), (10, 6), (10, 2), (10, 1), (11, 2), (9, 7), (9, 0), (11, 9)}
NM_CSV       = "results/name_movers/head_to_logits_causal_effect.csv"


def _path_patch_bnm(model, ioi, abc, means, nm_heads, metric, bnm_threshold):
    """Path patching in NM-ablated model to find backup name mover heads.

    For each candidate sender (sl, sh):
      - Both IOI and ABC runs have NM heads replaced by their per-example ABC means.
      - Sender z is patched from the ABC run.
      - BNM effect = metric(patched) - metric(nm_ablated_baseline).
      - Negative effect → sender contributes positively in NM-ablated model → BNM.
    """
    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    N        = len(ioi)
    seq      = ioi.toks.shape[1]

    nm_by_layer = {}
    for nl, nh in nm_heads:
        nm_by_layer.setdefault(nl, []).append(nh)

    nm_set = set(nm_heads)
    ioi_toks = ioi.toks.long()
    abc_toks = abc.toks.long()

    # Cache z on clean IOI and ABC runs.
    ioi_z = {}
    with model.trace({"input_ids": ioi_toks}):
        for l in range(n_layers):
            ioi_z[l] = model.transformer.h[l].attn.c_proj.input.save()

    abc_z = {}
    with model.trace({"input_ids": abc_toks}):
        for l in range(n_layers):
            abc_z[l] = model.transformer.h[l].attn.c_proj.input.save()

    def _build_z(sl, sh):
        """Return per-layer z tensors: NM heads ablated, sender patched from ABC."""
        out = {}
        for l in range(n_layers):
            z_h = ioi_z[l].reshape(N, seq, n_heads, d_head).clone()
            if l in nm_by_layer:
                for nh in nm_by_layer[l]:
                    z_h[:, :, nh, :] = means[l, :, :, nh, :]
            if l == sl and (sl, sh) not in nm_set:
                abc_z_h = abc_z[sl].reshape(N, seq, n_heads, d_head)
                z_h[:, :, sh, :] = abc_z_h[:, :, sh, :]
            out[l] = z_h.reshape(N, seq, n_heads * d_head)
        return out

    # NM-ablated baseline (no sender patch).
    baseline_z = _build_z(-1, -1)
    with model.trace({"input_ids": ioi_toks}):
        for l in range(n_layers):
            model.transformer.h[l].attn.c_proj.input[...] = baseline_z[l]
        nm_abl_logits = model.lm_head.output.save()
    nm_abl_m = metric(nm_abl_logits.cpu())
    del nm_abl_logits, baseline_z
    clear_cache()

    effects = np.zeros((n_layers, n_heads))
    for sl in range(n_layers):
        for sh in range(n_heads):
            if (sl, sh) in nm_set:
                continue
            patched_z = _build_z(sl, sh)
            with model.trace({"input_ids": ioi_toks}):
                for l in range(n_layers):
                    model.transformer.h[l].attn.c_proj.input[...] = patched_z[l]
                patched_logits = model.lm_head.output.save()
            effects[sl, sh] = metric(patched_logits.cpu()) - nm_abl_m
            del patched_logits, patched_z
            clear_cache()

    bnm = {(l, h) for l in range(n_layers) for h in range(n_heads)
           if (l, h) not in nm_set and effects[l, h] < -bnm_threshold}
    return bnm, effects


def run(nnm_threshold=0.2, bnm_threshold=0.05):
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    # NNM from precomputed NM path patching CSV.
    df  = pd.read_csv(NM_CSV)
    nnm = {(int(r['layer']), int(r['head'])) for _, r in df.iterrows()
           if r['causal_effect'] > nnm_threshold}
    print(f"NNM: {sorted(nnm)}  expected {sorted(EXPECTED_NNM)}")
    print("PASS" if EXPECTED_NNM <= nnm else f"WARNING missing {EXPECTED_NNM - nnm}")

    # BNM via path patching in NM-ablated model.
    print("\nBuilding IOI/ABC datasets and computing ABC means…")
    ioi  = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
    abc  = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc  = abc.gen_flipped_prompts(("S", "RAND"))
    means = compute_means(model, abc.toks.long(), abc.groups)

    N       = len(ioi)
    end_pos = ioi.word_idx["end"].long()

    def metric(logits):
        return logit_diff(
            logits[torch.arange(N), end_pos],
            ioi.io_tokenIDs, ioi.s_tokenIDs,
        ).mean().item()

    print("Running path patching in NM-ablated model…")
    bnm, effects = _path_patch_bnm(model, ioi, abc, means, NM_HEADS, metric, bnm_threshold)

    print(f"\nBNM: {sorted(bnm)}  expected {sorted(EXPECTED_BNM)}")
    missing = EXPECTED_BNM - bnm
    print("PASS" if not missing else f"WARNING missing {missing}")

    n_layers = effects.shape[0]
    n_heads  = effects.shape[1]
    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(np.abs(effects).max(), 1e-6)
    im = ax.imshow(effects, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head"); ax.set_ylabel("Layer")
    ax.set_title("Path patching effect in NM-ablated model\n(negative = BNM, positive = NNM-like)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs("plots/backup", exist_ok=True)
    plt.savefig("plots/backup/neg_backup.png", dpi=150); plt.close()

    os.makedirs("results/backup", exist_ok=True)
    with open("results/backup/all_nm_types.json", "w") as f:
        json.dump({"name_mover": [list(h) for h in NM_HEADS],
                   "negative":   sorted([list(h) for h in nnm]),
                   "backup":     sorted([list(h) for h in bnm])}, f, indent=2)
    return nnm, bnm


if __name__ == "__main__":
    run()
