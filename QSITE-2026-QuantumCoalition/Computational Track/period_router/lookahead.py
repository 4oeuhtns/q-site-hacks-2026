"""Gate-by-gate router with lookahead: the fallback for stretches that don't fit.

For a gate whose qubits are apart, every SWAP it makes moves one of them one
hop closer to the other, so routing a gate always ends. When several SWAPs do
that, it picks the one that leaves the next few gates closest together.
"""

from .chip import Chip
from .layout import Layout
from .program import upcoming_pairs

LOOKAHEAD = 20  # how many upcoming gates to consider
DECAY = 0.8  # each later gate counts this much less than the one before


def upcoming_cost(layout: Layout, upcoming: list[tuple[int, int, float]], chip: Chip) -> float:
    return sum(weight * chip.dist[layout.where[a]][layout.where[b]] for a, b, weight in upcoming)


def route_gate(layout: Layout, a: int, b: int, upcoming: list[tuple[int, int, float]], chip: Chip) -> list[tuple[int, int]]:
    """SWAPs that make logical a and b neighbours. Mutates `layout`."""
    swaps = []
    dist = chip.dist
    while dist[layout.where[a]][layout.where[b]] > 1:
        pa, pb = layout.where[a], layout.where[b]
        gap = dist[pa][pb]
        options = [
            (mover, step)
            for mover, other in ((pa, pb), (pb, pa))
            for step in sorted(chip.graph.neighbors(mover))
            if dist[step][other] == gap - 1
        ]

        def score(option: tuple[int, int]) -> tuple[float, tuple[int, int]]:
            layout.swap(*option)
            value = upcoming_cost(layout, upcoming, chip)
            layout.swap(*option)  # undo
            return value, option

        best = min(options, key=score)
        layout.swap(*best)
        swaps.append(best)
    return swaps


def route_span(program: list[tuple], start: int, end: int, layout: Layout, chip: Chip) -> list[tuple]:
    """Route program[start:end] gate by gate. Looks past `end` for lookahead. Mutates `layout`."""
    routed = []
    for index in range(start, end):
        op = program[index]
        if op[0] == "2Q":
            upcoming = upcoming_pairs(program, index + 1, LOOKAHEAD, DECAY)
            for p, q in route_gate(layout, op[1], op[2], upcoming, chip):
                routed.append(("SWAP", p, q))
        routed.append(layout.emit(op))
    return routed


def route_lookahead(program: list[tuple], placement: dict[int, int], chip: Chip) -> list[tuple]:
    """The whole program, gate by gate."""
    return route_span(program, 0, len(program), Layout(placement), chip)
