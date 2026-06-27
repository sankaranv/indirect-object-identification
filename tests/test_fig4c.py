import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fig4c_imports():
    from experiments.fig4c_si_combined import run
    assert callable(run)
