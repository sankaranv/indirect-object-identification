import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appH_imports():
    from experiments.appendix.appH_head_copy_strength import compute_copy_scores
    from experiments.appendix.appHI_induction_pattern_scores import compute_all_scores

    assert callable(compute_copy_scores)
    assert callable(compute_all_scores)
