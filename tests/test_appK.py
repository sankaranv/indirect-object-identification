import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appK_saves_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/circuit", exist_ok=True)
    from experiments.appK_minimality_sets import run

    run()
    import csv

    with open("results/circuit/fig20_k_sets.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 26
    assert "head" in rows[0] and "head_type" in rows[0] and "K_set" in rows[0]
