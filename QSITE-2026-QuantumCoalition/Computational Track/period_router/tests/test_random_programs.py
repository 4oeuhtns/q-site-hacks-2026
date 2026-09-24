import networkx as nx
import pytest

from period_router.random_programs import STYLES, random_program, unseen_chips


@pytest.mark.parametrize("style", STYLES)
def test_program_shape(style):
    program = random_program(8, 30, style, seed=1)
    two_qubit = [op for op in program if op[0] == "2Q"]
    assert len(two_qubit) == 30
    assert all(op[1] != op[2] for op in two_qubit)
    assert all(0 <= q < 8 for op in program for q in op[1:])
    assert program == random_program(8, 30, style, seed=1)  # same seed, same program


def test_unknown_style_is_an_error():
    with pytest.raises(ValueError):
        random_program(8, 30, "sideways", seed=1)


def test_unseen_chips_are_connected_and_20_qubits():
    for name, graph in unseen_chips().items():
        assert graph.number_of_nodes() == 20, name
        assert nx.is_connected(graph), name
