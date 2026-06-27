"""Appendix E / Fig 14: Save IOI dataset templates to CSV."""
import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))

from ioi_dataset import BABA_TEMPLATES, ABBA_TEMPLATES


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
