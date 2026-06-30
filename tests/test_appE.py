import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appE_saves_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results/templates", exist_ok=True)
    from experiments.appendix.appE_templates import run

    run()
    import csv

    with open("results/templates/fig14.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert "template" in rows[0]
    assert "pattern" in rows[0]
