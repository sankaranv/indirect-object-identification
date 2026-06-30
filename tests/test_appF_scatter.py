import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appF_scatter_imports():
    from experiments.appendix.appF_backup_name_mover_discovery import run

    assert callable(run)
