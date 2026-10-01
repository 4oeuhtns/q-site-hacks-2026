"""Facts about the hardware graph, computed once per solve() call.

Everything the router needs to know about the chip is derived here from the
graph it is given, so nothing in the package is specific to the 20-qubit chip.
"""

from dataclasses import dataclass

import networkx as nx


@dataclass
class Chip:
    graph: nx.Graph
    dist: dict  # dist[p][q] = number of hops between physical qubits p and q
    max_degree: int  # most neighbours any physical qubit has
    girth: float  # length of the shortest loop (inf if the graph has none)
    bipartite: bool  # True if every loop has even length
    center: int  # physical qubit closest to everything; used to seed placement
    neighbors: dict  # neighbors[p] = frozenset of p's neighbours (plain dict: fast in hot loops)


def make_chip(graph: nx.Graph) -> Chip:
    if not nx.is_connected(graph):
        raise ValueError("hardware graph must be connected")
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    return Chip(
        graph=graph,
        neighbors={p: frozenset(graph.neighbors(p)) for p in graph.nodes},
        dist=dist,
        max_degree=max(degree for _, degree in graph.degree()),
        girth=nx.girth(graph),
        bipartite=nx.is_bipartite(graph),
        center=min(graph.nodes, key=lambda p: (max(dist[p].values()), -graph.degree(p), p)),
    )
