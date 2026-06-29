"""Sorted bar chart of head→logit path-patching causal effects, colored by circuit role."""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

HEAD_TYPE_COLORS = {
    "name_mover": "#2196F3",
    "negative_name_mover": "#F44336",
    "backup_name_mover": "#64B5F6",
    "s2_inhibition": "#FF9800",
    "induction": "#9C27B0",
    "duplicate_token": "#4CAF50",
    "previous_token": "#8BC34A",
    "other": "#BDBDBD",
}

CIRCUIT_HEADS = {
    "name_mover": [(9, 9), (10, 0), (9, 6)],
    "negative_name_mover": [(10, 7), (11, 10)],
    "backup_name_mover": [
        (10, 10),
        (10, 6),
        (10, 2),
        (10, 1),
        (11, 2),
        (9, 7),
        (9, 0),
        (11, 9),
    ],
    "s2_inhibition": [(7, 3), (7, 9), (8, 6), (8, 10)],
    "induction": [(5, 5), (5, 8), (5, 9), (6, 9)],
    "duplicate_token": [(0, 1), (0, 10), (3, 0)],
    "previous_token": [(2, 2), (4, 11)],
}


def _head_type(layer, head):
    for t, heads in CIRCUIT_HEADS.items():
        if (layer, head) in heads:
            return t
    return "other"


def run():
    df = pd.read_csv("results/name_movers/head_to_logits_causal_effect.csv")
    df["head_type"] = df.apply(
        lambda r: _head_type(int(r["layer"]), int(r["head"])), axis=1
    )
    df["abs_effect"] = df["causal_effect"].abs()
    df = df.sort_values("abs_effect", ascending=False).reset_index(drop=True)
    df["label"] = df.apply(lambda r: f"{int(r['layer'])}.{int(r['head'])}", axis=1)

    colors = [HEAD_TYPE_COLORS[t] for t in df["head_type"]]

    fig, ax = plt.subplots(figsize=(max(16, len(df) * 0.18), 5))
    ax.bar(range(len(df)), df["causal_effect"].values, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"].values, rotation=90, fontsize=6)
    ax.set_ylabel("Causal effect (patched − clean logit diff)")
    ax.set_title("Head → logit causal effect, sorted by absolute magnitude")

    patches = [
        mpatches.Patch(color=c, label=t.replace("_", " "))
        for t, c in HEAD_TYPE_COLORS.items()
        if t != "other"
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=7)

    plt.tight_layout()
    os.makedirs("plots/name_movers", exist_ok=True)
    plt.savefig("plots/name_movers/fig3b.png", dpi=150)
    plt.close()
    print("Saved plots/name_movers/fig3b.png")


if __name__ == "__main__":
    run()
