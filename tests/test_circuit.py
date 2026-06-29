import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_circuit_head_count():
    from circuit import CIRCUIT

    heads = [h for v in CIRCUIT.values() for h in v]
    assert len(heads) == len(set(heads)), "Duplicate heads"
    assert len(heads) == 26, f"Expected 26, got {len(heads)}"


def test_circuit_types():
    from circuit import CIRCUIT

    assert set(CIRCUIT) == {
        "name_mover",
        "backup_name_mover",
        "negative_name_mover",
        "s2_inhibition",
        "induction",
        "duplicate_token",
        "previous_token",
    }


def test_k_covers_all_circuit_heads():
    from circuit import CIRCUIT, K_FOR_EACH_COMPONENT

    all_heads = {h for v in CIRCUIT.values() for h in v}
    assert all_heads == set(K_FOR_EACH_COMPONENT)
