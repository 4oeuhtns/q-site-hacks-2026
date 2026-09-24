import random

from starter_kit import build_hardware_graph

from period_router.chip import make_chip
from period_router.layout import Layout
from period_router.random_programs import unseen_chips
from period_router.token_swap import token_swap


def test_worked_example(chip):
    layout = Layout({0: 5, 1: 6, 2: 7, 3: 11})
    swaps = token_swap(layout, {0: 7, 2: 6, 3: 11}, chip)
    assert swaps == [(5, 6), (6, 7)]
    assert layout.where == {0: 7, 1: 5, 2: 6, 3: 11}


def test_nothing_to_do(chip):
    layout = Layout({0: 5, 1: 6})
    assert token_swap(layout, {0: 5}, chip) == []


def test_random_reshuffles_always_arrive_using_real_wires():
    chips = {"challenge": build_hardware_graph(), **unseen_chips()}
    rng = random.Random(0)
    for graph in chips.values():
        chip = make_chip(graph)
        chairs = list(graph.nodes)
        for _ in range(200):
            count = rng.randint(1, len(chairs))
            qubits = rng.sample(range(40), count)
            layout = Layout(dict(zip(qubits, rng.sample(chairs, count))))
            movers = rng.sample(qubits, rng.randint(1, count))
            target = dict(zip(movers, rng.sample(chairs, len(movers))))
            swaps = token_swap(layout, target, chip)
            assert all(graph.has_edge(p, q) for p, q in swaps)
            assert all(layout.where[q] == chair for q, chair in target.items())
