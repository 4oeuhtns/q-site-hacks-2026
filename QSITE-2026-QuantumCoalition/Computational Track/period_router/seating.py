"""Seatings: where to put a period's qubits so every interacting pair are neighbours.

In graph terms a seating is a subgraph monomorphism from the period's
interaction graph into the chip. We find them with our own backtracking search
rather than networkx's VF2, because ours has a step budget: VF2 can take
exponential time to prove that a large pattern does NOT fit, and cannot be
interrupted.

When choosing among seatings, the cost of a seating is
    (total hops the period's qubits move from where they sit now)
  + UPCOMING_WEIGHT x (how far apart the next gates' pairs would end up)
The second term is the look-ahead across periods: prefer seatings that are
cheap to leave, not just cheap to reach.
"""

import math
from collections import defaultdict

import networkx as nx

from .chip import Chip

FIT_BUDGET = 20_000  # steps to spend proving a seating exists
CLOSEST_BUDGET = 2_000  # steps to spend improving on the first seating found
UPCOMING_WEIGHT = 1.0  # one hop of future distance counts the same as one hop moved now


def search_order(pattern: nx.Graph) -> list[int]:
    """Breadth-first order, so every qubit after the first in its group has a
    partner already seated. That caps its options at that chair's neighbours."""
    order = []
    components = sorted(nx.connected_components(pattern), key=lambda c: (-len(c), min(c)))
    for component in components:
        root = max(sorted(component), key=pattern.degree)
        order.append(root)
        order.extend(child for _, child in nx.bfs_edges(pattern, root))
    return order


def search_seating(
    pattern: nx.Graph,
    chip: Chip,
    reference: dict[int, int] | None = None,
    upcoming: list[tuple[int, int, float]] = (),
    budget: int = FIT_BUDGET,
) -> dict[int, int] | None:
    """Find a seating (logical -> physical) for every node of `pattern`.

    reference=None: return the first seating found.
    reference given: return the cheapest seating found (see module docstring).
    upcoming: (a, b, weight) pairs for the gates after this period.
    Returns None if the budget runs out before any seating is found.
    """
    order = search_order(pattern)
    dist, neighbors = chip.dist, chip.neighbors
    partners = {node: list(pattern.neighbors(node)) for node in order}
    need = {node: len(partners[node]) for node in order}
    all_chairs = frozenset(neighbors)
    seat: dict[int, int] = {}
    used: set[int] = set()
    best_cost = math.inf
    best_seat: dict[int, int] | None = None
    steps = 0

    pairs_of = defaultdict(list)  # node -> [(other qubit, weight)]
    for a, b, weight in upcoming:
        if a in pattern:
            pairs_of[a].append((b, weight))
        if b in pattern:
            pairs_of[b].append((a, weight))

    def added_cost(node: int, chair: int) -> float:
        """Cost of seating `node` on `chair`, given who is seated already."""
        if reference is None:
            return 0.0
        total = float(dist[reference[node]][chair])
        for other, weight in pairs_of[node]:
            if other in seat:  # both in the period: count once, when the second is seated
                total += UPCOMING_WEIGHT * weight * (dist[chair][seat[other]] - 1)
            elif other not in pattern and other in reference:  # other stays where it is
                total += UPCOMING_WEIGHT * weight * max(0, dist[chair][reference[other]] - 1)
        return total

    def candidates(node: int) -> list[tuple[float, int]]:
        seated_partners = [seat[n] for n in partners[node] if n in seat]
        if seated_partners:
            options = neighbors[seated_partners[0]]
            for chair in seated_partners[1:]:
                options = options & neighbors[chair]
        else:
            options = all_chairs
        scored = [(added_cost(node, c), c) for c in options if c not in used and len(neighbors[c]) >= need[node]]
        return sorted(scored)

    def extend(k: int, cost: float) -> bool:
        """Returns True when the search should stop."""
        nonlocal best_cost, best_seat, steps
        if k == len(order):
            best_cost, best_seat = cost, dict(seat)
            return reference is None or cost == 0  # nothing can beat zero
        node = order[k]
        for extra, chair in candidates(node):
            new_cost = cost + extra
            if new_cost >= best_cost:
                continue
            steps += 1
            if steps > budget:
                return True
            seat[node] = chair
            used.add(chair)
            stop = extend(k + 1, new_cost)
            del seat[node]
            used.discard(chair)
            if stop:
                return True
        return False

    extend(0, 0.0)
    return best_seat


def closest_seating(
    pattern: nx.Graph,
    reference: dict[int, int],
    chip: Chip,
    upcoming: list[tuple[int, int, float]] = (),
) -> dict[int, int]:
    """The cheapest seating found (see module docstring). The pattern must be one
    that fit.fits() accepted, so a seating is known to exist."""
    seat = search_seating(pattern, chip, reference, upcoming, CLOSEST_BUDGET) or search_seating(pattern, chip)
    if seat is None:
        raise ValueError("pattern has no seating; check fits() first")
    return seat
