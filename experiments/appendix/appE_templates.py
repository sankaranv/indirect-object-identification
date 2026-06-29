"""Appendix E / Fig 14: Save IOI dataset templates to CSV."""

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ioi_dataset import ABBA_TEMPLATES, BABA_TEMPLATES


def run():
    os.makedirs("results/templates", exist_ok=True)
    rows = []
    for tmpl in BABA_TEMPLATES:
        rows.append({"template": tmpl, "pattern": "BABA"})
    for tmpl in ABBA_TEMPLATES:
        rows.append({"template": tmpl, "pattern": "ABBA"})

    with open("results/templates/fig14.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["template", "pattern"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved results/templates/fig14.csv ({len(rows)} templates)")


if __name__ == "__main__":
    run()
