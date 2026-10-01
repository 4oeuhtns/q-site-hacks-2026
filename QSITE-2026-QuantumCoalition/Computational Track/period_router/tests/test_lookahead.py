from starter_kit import BENCHMARKS, score_summary

from period_router.layout import Layout
from period_router.lookahead import route_gate, route_lookahead


def test_route_gate_needs_exactly_distance_minus_one_swaps(chip):
    layout = Layout({0: 0, 1: 19})  # 7 hops apart
    swaps = route_gate(layout, 0, 1, [], chip)
    assert len(swaps) == 6
    assert all(q in chip.neighbors[p] for p, q in swaps)
    assert chip.dist[layout.where[0]][layout.where[1]] == 1


def test_route_gate_does_nothing_for_neighbours(chip):
    layout = Layout({0: 5, 1: 6})
    assert route_gate(layout, 0, 1, [], chip) == []


def test_route_lookahead_is_valid_on_every_benchmark(hw, chip):
    for name, program in BENCHMARKS.items():
        placement = {q: q for q in {q for op in program for q in op[1:]}}
        routed = route_lookahead(program, placement, chip)
        result = score_summary(program, hw, placement, routed)
        assert result["valid"], (name, result["message"])
