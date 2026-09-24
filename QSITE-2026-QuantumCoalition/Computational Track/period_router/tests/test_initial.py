import pytest

from starter_kit import BENCHMARKS

from period_router.initial import initial_placement, label_order_placement
from period_router.program import logical_qubits


def test_every_used_qubit_gets_its_own_chair(chip):
    for name, program in BENCHMARKS.items():
        placement = initial_placement(program, chip)
        assert set(placement) == set(logical_qubits(program)), name
        assert len(set(placement.values())) == len(placement), name
        assert set(placement.values()) <= set(chip.neighbors), name


def test_missing_labels_are_not_placed(chip):
    # qaoa_random never uses qubits 4 and 7; placing them makes the scorer reject the answer
    placement = initial_placement(BENCHMARKS["qaoa_random"], chip)
    assert 4 not in placement and 7 not in placement


def test_first_period_is_seated_with_every_pair_adjacent(chip):
    placement = initial_placement(BENCHMARKS["chain_trotter"], chip)
    for _, a, b in BENCHMARKS["chain_trotter"]:
        assert placement[b] in chip.neighbors[placement[a]]


def test_too_many_qubits_is_an_error(chip):
    program = [("2Q", q, q + 1) for q in range(21)]  # 22 qubits on a 20-qubit chip
    with pytest.raises(ValueError):
        initial_placement(program, chip)


def test_label_order_placement(chip):
    assert label_order_placement([("2Q", 3, 7), ("1Q", 9)], chip) == {3: 0, 7: 1, 9: 2}
