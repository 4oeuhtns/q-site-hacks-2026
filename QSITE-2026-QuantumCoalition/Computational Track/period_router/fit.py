"""Does a stretch of the program fit on the chip with zero SWAPs?

"Fits" means: there is a seating where every interacting pair sits on
neighbouring physical qubits. In graph terms, the interaction graph is
subgraph-monomorphic to the chip. Monomorphic, not isomorphic: two qubits that
never interact are allowed to sit next to each other.
"""

import networkx as nx

from .chip import Chip
from .program import interaction_graph
from .seating import search_seating


def cannot_fit(pattern: nx.Graph, chip: Chip) -> bool:
    """Instant proofs that a pattern cannot fit. False means 'not ruled out'."""
    if pattern.number_of_edges() == 0:
        return False
    if pattern.number_of_nodes() > chip.graph.number_of_nodes():
        return True
    if max(degree for _, degree in pattern.degree()) > chip.max_degree:
        return True  # someone has more partners than any chair has neighbours
    if chip.bipartite and not nx.is_bipartite(pattern):
        return True  # an odd loop cannot be drawn on a chip with only even loops
    if nx.girth(pattern) < chip.girth:
        return True  # a loop shorter than the chip's shortest loop
    return False


def fits(pattern: nx.Graph, chip: Chip) -> bool:
    """True only if a seating was actually found. If the search budget runs out,
    the answer is False: the period just ends earlier, which is always safe."""
    if pattern.number_of_edges() == 0:
        return True
    if cannot_fit(pattern, chip):
        return False
    return search_seating(pattern, chip) is not None


def longest_fitting_end(program: list[tuple], start: int, chip: Chip) -> int:
    """Largest end such that program[start:end] fits. Always at least start + 1.

    Binary search works because fitting is monotone: if a stretch fits, every
    shorter stretch from the same start fits too. It only ever moves `lo` onto
    an end that fits(), so the answer is always a stretch that really fits.
    """
    lo, hi = start + 1, len(program)  # program[start:lo] is a single op, which always fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(interaction_graph(program[start:mid]), chip):
            lo = mid
        else:
            hi = mid - 1
    return lo
