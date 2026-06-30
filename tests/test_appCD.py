import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appC_imports():
    from experiments.appendix.appC_s2_inhibition_key_signals import run as run_c
    from experiments.appendix.appD_induction_key_signals import run as run_d

    assert callable(run_c) and callable(run_d)
