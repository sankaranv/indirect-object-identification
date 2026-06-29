import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_patching_result_matrix():
    from patching import PatchingResult

    scores = {
        (layer, head): float(layer * 4 + head)
        for layer in range(3)
        for head in range(4)
    }
    r = PatchingResult(scores=scores, n_layers=3, n_heads=4)
    mat = r.as_matrix()
    assert mat.shape == (3, 4)
    assert mat[1, 2] == scores[(1, 2)]


def test_patching_result_top_k():
    from patching import PatchingResult

    scores = {(0, 0): 1.0, (0, 1): 3.0, (1, 0): 2.0}
    r = PatchingResult(scores=scores, n_layers=2, n_heads=2)
    assert r.top_k(2) == [(0, 1), (1, 0)]


def test_imports():
    from patching import path_patch_head_to_logits, path_patch_head_to_heads

    assert callable(path_patch_head_to_logits)
    assert callable(path_patch_head_to_heads)
