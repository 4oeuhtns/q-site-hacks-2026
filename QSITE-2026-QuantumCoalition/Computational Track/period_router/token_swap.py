"""Reshuffle: neighbour SWAPs that move some qubits onto target chairs.

Qubits without a target, and empty chairs, can end up anywhere.

Phase 1 is greedy: keep making the SWAP that brings target-holders closest to
their targets in total, while some SWAP still helps. It is short but can get
stuck, e.g. when a qubit must pass through one that is already in place.

Phase 2 always finishes: fill the leaves of a spanning tree one at a time, then
lock them. It is longer, so it only runs when phase 1 gets stuck. Measured: the
reshuffles the period router actually keeps are almost all small (0-3 SWAPs),
so a cleverer phase 2 would barely change the score.
"""

import networkx as nx

from .chip import Chip
from .layout import Layout


def token_swap(layout: Layout, target: dict[int, int], chip: Chip) -> list[tuple[int, int]]:
    """Move every logical qubit in `target` onto its chair. Mutates `layout`
    and returns the SWAPs made, in order."""
    swaps: list[tuple[int, int]] = []
    dist = chip.dist

    def swap(p: int, q: int) -> None:
        if layout.occupant.get(p) is None and layout.occupant.get(q) is None:
            return  # swapping two empty chairs changes nothing
        layout.swap(p, q)
        swaps.append((p, q))

    def gain(p: int, q: int) -> int:
        a, b = layout.occupant.get(p), layout.occupant.get(q)
        total = 0
        if a in target:
            total += dist[p][target[a]] - dist[q][target[a]]
        if b in target:
            total += dist[q][target[b]] - dist[p][target[b]]
        return total

    # Phase 1: greedy
    while True:
        best_edge, best_gain = None, 0
        for p, q in chip.graph.edges:
            g = gain(p, q)
            if g > best_gain:
                best_edge, best_gain = (p, q), g
        if best_edge is None:
            break
        swap(*best_edge)

    if all(layout.where[logical] == chair for logical, chair in target.items()):
        return swaps

    # Phase 2: spanning-tree leaf filling
    remaining = nx.Graph(nx.bfs_tree(chip.graph, chip.center))
    wanted_at = {chair: logical for logical, chair in target.items()}
    while remaining.number_of_nodes():
        leaf = min(v for v in remaining if remaining.degree(v) <= 1)
        wanted = wanted_at.get(leaf)
        if wanted is not None:
            source = layout.where[wanted]
        else:  # the leaf needs anything that has no target: an empty chair or a free qubit
            free = [v for v in remaining if layout.occupant.get(v) not in target]
            lengths = nx.single_source_shortest_path_length(remaining, leaf)
            source = min(free, key=lambda v: (lengths[v], v))
        path = nx.shortest_path(remaining, source, leaf)
        for p, q in zip(path, path[1:]):
            swap(p, q)
        remaining.remove_node(leaf)
    return swaps
