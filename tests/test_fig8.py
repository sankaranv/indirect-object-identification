import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fig8_imports():
    from experiments.fig8_performance_summary import build_adversarial_dataset, run
    assert callable(build_adversarial_dataset)
    assert callable(run)
