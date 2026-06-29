import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appM_imports():
    from experiments.appM_greedy_completeness import greedy_k_sample, run

    assert callable(greedy_k_sample)
    assert callable(run)
