import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appF_bars_imports():
    from experiments.appendix.appF_backup_name_mover_effects import run

    assert callable(run)
