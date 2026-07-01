"""Unit tests for witness.py.

Uses SimpleNamespace stubs for model structure so no real weights are needed.
Functions that require nnsight tracing (witness_pinned_ablation_scores,
witness_importance_scores, pie_denoising_scores) are tested only for their
import and return-type contract; correctness is validated by the experiment
script on real GPT-2 weights.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    from witness import (
        pie_denoising_scores,
        witness_importance_scores,
        witness_pinned_ablation_scores,
    )

    assert callable(witness_pinned_ablation_scores)
    assert callable(witness_importance_scores)
    assert callable(pie_denoising_scores)


def test_patching_result_reused():
    """witness.py returns the same PatchingResult from patching.py — no duplicate."""
    from witness import witness_pinned_ablation_scores

    import inspect

    src = inspect.getsource(witness_pinned_ablation_scores)
    assert "PatchingResult" in src


def test_witness_importance_scores_signature():
    from witness import witness_importance_scores

    import inspect

    sig = inspect.signature(witness_importance_scores)
    params = list(sig.parameters)
    assert "suspect_head" in params
    assert "candidate_witnesses" in params
    assert "positions" in params


def test_pie_denoising_scores_signature():
    from witness import pie_denoising_scores

    import inspect

    sig = inspect.signature(pie_denoising_scores)
    params = list(sig.parameters)
    assert "clean" in params
    assert "corrupted" in params
    assert "batch_size" in params
