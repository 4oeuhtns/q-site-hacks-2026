from collections import Counter

from period_router.program import (
    first_appearance,
    interaction_graph,
    logical_qubits,
    partner_counts,
    upcoming_pairs,
)

PROGRAM = [("2Q", 0, 1), ("1Q", 5), ("2Q", 1, 2), ("2Q", 0, 1)]


def test_logical_qubits_allows_gaps_in_labels():
    assert logical_qubits([("2Q", 3, 7), ("1Q", 9)]) == [3, 7, 9]


def test_first_appearance():
    assert first_appearance(PROGRAM) == [0, 1, 5, 2]


def test_interaction_graph_ignores_1q_gates_and_repeats():
    graph = interaction_graph(PROGRAM)
    assert sorted(graph.edges) == [(0, 1), (1, 2)]
    assert 5 not in graph


def test_upcoming_pairs_weights_decay():
    assert upcoming_pairs(PROGRAM, 1, limit=2, decay=0.5) == [(1, 2, 1.0), (0, 1, 0.5)]
    assert upcoming_pairs(PROGRAM, 4, limit=5, decay=0.5) == []


def test_partner_counts():
    counts = partner_counts(PROGRAM)
    assert counts[0] == Counter({1: 2})
    assert counts[1] == Counter({0: 2, 2: 1})
