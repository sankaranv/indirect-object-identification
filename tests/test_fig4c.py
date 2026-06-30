import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_fig4c_imports():
    from experiments.discovery.fig4c_s2_inhibition_combined import run

    assert callable(run)
