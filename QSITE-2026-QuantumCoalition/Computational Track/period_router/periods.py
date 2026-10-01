"""The period router.

Cut the program into periods: the longest stretches that fit with zero SWAPs.
For each period, compare two ways of handling it and keep the cheaper:
  reseat - reshuffle to the closest seating where the whole period fits, or
  route  - leave the seating alone and route the period gate by gate.
"Cheaper" is the real score (SWAPs + 0.5 x depth) of everything emitted so far.
"""

from solver.cost import cost

from .chip import Chip
from .fit import longest_fitting_end
from .layout import Layout
from .lookahead import DECAY, LOOKAHEAD, route_span
from .program import interaction_graph, upcoming_pairs
from .seating import closest_seating
from .token_swap import token_swap


def reseat_span(program: list[tuple], start: int, end: int, layout: Layout, chip: Chip) -> list[tuple]:
    """Reshuffle to the closest seating that fits program[start:end], then emit it. Mutates `layout`."""
    pattern = interaction_graph(program[start:end])
    routed = []
    if pattern.number_of_edges():
        upcoming = upcoming_pairs(program, end, LOOKAHEAD, DECAY)
        target = closest_seating(pattern, layout.where, chip, upcoming)
        routed = [("SWAP", p, q) for p, q in token_swap(layout, target, chip)]
    routed.extend(layout.emit(op) for op in program[start:end])
    return routed


def route_periods(program: list[tuple], placement: dict[int, int], chip: Chip, stats: dict | None = None) -> list[tuple]:
    layout = Layout(placement)
    routed: list[tuple] = []
    start = 0
    while start < len(program):
        end = longest_fitting_end(program, start, chip)

        reseat_layout = layout.copy()
        reseated = reseat_span(program, start, end, reseat_layout, chip)
        route_layout = layout.copy()
        gate_by_gate = route_span(program, start, end, route_layout, chip)

        if cost(routed + reseated) <= cost(routed + gate_by_gate):
            routed += reseated
            layout = reseat_layout
            choice = "reseat"
        else:
            routed += gate_by_gate
            layout = route_layout
            choice = "route"

        if stats is not None:
            stats.setdefault("periods", []).append((end - start, choice))
        start = end
    return routed
