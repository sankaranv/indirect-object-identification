import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appJ_imports():
    from experiments.appJ_mlp_knockout import run
    assert callable(run)
