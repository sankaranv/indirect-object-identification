import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fig3b_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import csv
    import os

    os.makedirs("results/name_movers", exist_ok=True)
    # Minimal stub CSV
    with open(
        "results/name_movers/head_to_logits_causal_effect.csv", "w", newline=""
    ) as f:
        w = csv.DictWriter(f, fieldnames=["layer", "head", "causal_effect"])
        w.writeheader()
        for layer in range(3):
            for head in range(4):
                w.writerow(
                    {
                        "layer": layer,
                        "head": head,
                        "causal_effect": (layer * 4 + head - 6) * 0.1,
                    }
                )
    os.makedirs("plots/name_movers", exist_ok=True)
    from experiments.fig3b_causal_effect_bar import run

    run()
    assert os.path.exists("plots/name_movers/fig3b.png")
