import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appF_bars_imports():
    from experiments.appF_bnm_bars import run
    assert callable(run)
