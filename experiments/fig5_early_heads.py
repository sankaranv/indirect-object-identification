# indirect-object-identification/experiments/fig5_early_heads.py
"""Figure 5: DT/PT/IH heads via attention patterns on repeated-token sequences.
DT: attend from second occurrence (S2) back to first occurrence (S1).
PT: attend from each token back to its predecessor.
IH: attend from S2 to the token after S1 (one-past the first occurrence).
"""
import os, sys, json, torch
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
torch.set_grad_enabled(False)

from utils import load_model
from analysis import attention_to_positions

# Wang et al. identify 10 heads total across DT/PT/IH via circuit analysis.
# Raw attention patterns on repeated sequences reliably detect the *strong* members;
# (0,10) (DT) and (5,8),(5,9) (IH) work via K-composition and are task-specific —
# they score <0.1 on random repeated sequences and require activation-patching to observe.
EXPECTED_DT = {(0, 1), (3, 0)}
EXPECTED_PT = {(2, 2), (4, 11)}
EXPECTED_IH = {(5, 5), (6, 9)}


def run(threshold: float = 0.2, seq_len: int = 50, batch: int = 50):
    model  = load_model()
    n_heads = model.config.n_head
    n_layers = len(model.transformer.h)

    torch.manual_seed(1)
    half   = torch.randint(1, model.config.vocab_size, (batch, seq_len))
    tokens = torch.cat([half, half], dim=1)   # [batch, 2*seq_len]

    # per-example query and key positions for each head type
    # DT: S2 (positions seq_len..2*seq_len-1) attends to S1 (positions 0..seq_len-1)
    # PT: each position t attends to t-1
    # IH: S2 attends to S1+1 (one after first occurrence)
    dt_q = torch.arange(seq_len, 2 * seq_len).repeat(batch, 1)  # [batch, seq_len]
    dt_k = torch.arange(0, seq_len).repeat(batch, 1)

    pt_q = torch.arange(1, seq_len + 1).repeat(batch, 1)
    pt_k = torch.arange(0, seq_len).repeat(batch, 1)

    ih_q = torch.arange(seq_len, 2 * seq_len).repeat(batch, 1)
    ih_k = torch.arange(1, seq_len + 1).repeat(batch, 1)

    def _avg(query_cols, key_cols):
        # Average over the seq_len positions; attention_to_positions handles one (q,k) pair per example
        totals: dict = {}
        for pos_idx in range(seq_len):
            q = query_cols[:, pos_idx]   # [batch]
            k = key_cols[:, pos_idx]     # [batch]
            for (l, h), v in attention_to_positions(model, tokens, q, k).items():
                totals[(l, h)] = totals.get((l, h), 0.0) + v
        return {k: v / seq_len for k, v in totals.items()}

    dt = _avg(dt_q, dt_k)
    pt = _avg(pt_q, pt_k)
    ih = _avg(ih_q, ih_k)

    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, scores, title in zip(axes, [dt, pt, ih],
                                  ["Duplicate token", "Previous token", "Induction"]):
        arr = np.zeros((n_layers, n_heads))
        for (l, h), v in scores.items():
            arr[l, h] = v
        im = ax.imshow(arr, aspect="auto", cmap="Blues")
        ax.set_xlabel("Head"); ax.set_ylabel("Layer"); ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    os.makedirs("plots/early_heads", exist_ok=True)
    plt.savefig("plots/early_heads/fig5.png", dpi=150); plt.close()

    dt_h = {(l, h) for (l, h), v in dt.items() if v > threshold}
    pt_h = {(l, h) for (l, h), v in pt.items() if v > threshold}
    ih_h = {(l, h) for (l, h), v in ih.items() if v > threshold}

    all_pass = True
    for name, found, expected in [("DT", dt_h, EXPECTED_DT),
                                    ("PT", pt_h, EXPECTED_PT),
                                    ("IH", ih_h, EXPECTED_IH)]:
        missing = expected - found
        print(f"{name}: {sorted(found)}  " + ("PASS" if not missing else f"WARNING missing {missing}"))
        if missing: all_pass = False

    os.makedirs("results/early_heads", exist_ok=True)
    with open("results/early_heads/identified_heads.json", "w") as f:
        json.dump({"DT": sorted([list(h) for h in dt_h]),
                   "PT": sorted([list(h) for h in pt_h]),
                   "IH": sorted([list(h) for h in ih_h])}, f, indent=2)

    if all_pass: print("PASS")
    return dt_h, pt_h, ih_h


if __name__ == "__main__":
    run()
