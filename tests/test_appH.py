import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appH_imports():
    from experiments.appH_copy_scores import compute_copy_scores
    from experiments.appHI_dup_scores import compute_all_scores

    assert callable(compute_copy_scores)
    assert callable(compute_all_scores)
