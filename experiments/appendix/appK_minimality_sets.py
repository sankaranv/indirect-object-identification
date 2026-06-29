"""Appendix K / Fig 20: Export minimality companion sets to CSV."""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from circuit import CIRCUIT, K_FOR_EACH_COMPONENT


def run():
    head_to_type = {lh: t for t, heads in CIRCUIT.items() for lh in heads}
    os.makedirs("results/circuit", exist_ok=True)
    with open("results/circuit/fig20_k_sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "head_type", "K_set", "K_size"])
        w.writeheader()
        for head, K in K_FOR_EACH_COMPONENT.items():
            w.writerow(
                {
                    "head": str(head),
                    "head_type": head_to_type.get(head, "unknown"),
                    "K_set": str(sorted(K)),
                    "K_size": len(K),
                }
            )
    print(f"Saved results/circuit/fig20_k_sets.csv ({len(K_FOR_EACH_COMPONENT)} heads)")


if __name__ == "__main__":
    run()
