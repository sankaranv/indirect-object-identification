import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appA_imports():
    from experiments.appA_signal_decomposition import build_counterfactual_datasets, run

    assert callable(build_counterfactual_datasets)
    assert callable(run)
