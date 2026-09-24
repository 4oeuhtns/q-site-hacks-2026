import networkx as nx
import pytest

from period_router.chip import make_chip


def test_challenge_chip_facts(chip):
    assert chip.max_degree == 3
    assert chip.girth == 6
    assert chip.bipartite is True
    assert chip.center == 9
    assert chip.dist[0][19] == 7
    assert chip.neighbors[5] == frozenset({4, 6, 9})


def test_facts_come_from_the_graph_not_the_challenge():
    ring = make_chip(nx.cycle_graph(7))
    assert ring.max_degree == 2
    assert ring.girth == 7
    assert ring.bipartite is False  # odd ring


def test_disconnected_graph_is_rejected():
    with pytest.raises(ValueError):
        make_chip(nx.Graph([(0, 1), (2, 3)]))
