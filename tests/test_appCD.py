import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_appC_imports():
    from experiments.appC_si_keys import run as run_c
    from experiments.appD_ih_keys import run as run_d

    assert callable(run_c) and callable(run_d)
