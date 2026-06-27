import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_fig3b_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import csv, os
    os.makedirs("results/name_movers", exist_ok=True)
    # Minimal stub CSV
    with open("results/name_movers/head_to_logits_causal_effect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer","head","causal_effect"])
        w.writeheader()
        for l in range(3):
            for h in range(4):
                w.writerow({"layer": l, "head": h, "causal_effect": (l*4+h - 6)*0.1})
    os.makedirs("plots/name_movers", exist_ok=True)
    from experiments.fig3b_causal_effect_bar import run
    run()
    assert os.path.exists("plots/name_movers/fig3b.png")
