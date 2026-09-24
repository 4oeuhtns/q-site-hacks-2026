"""Helpers for reading a program: which qubits it uses and who talks to whom."""

from collections import Counter, defaultdict

import networkx as nx


def logical_qubits(program: list[tuple]) -> list[int]:
    """Every logical qubit the program mentions, sorted. Labels can have gaps."""
    return sorted({qubit for op in program for qubit in op[1:]})


def first_appearance(program: list[tuple]) -> list[int]:
    """Logical qubits in the order they first appear in the program."""
    seen: dict[int, None] = {}
    for op in program:
        for qubit in op[1:]:
            seen.setdefault(qubit, None)
    return list(seen)


def interaction_graph(ops: list[tuple]) -> nx.Graph:
    """One node per logical qubit in a 2Q gate, one edge per pair that interacts."""
    graph = nx.Graph()
    for op in ops:
        if op[0] == "2Q":
            graph.add_edge(op[1], op[2])
    return graph


def upcoming_pairs(program: list[tuple], start: int, limit: int, decay: float) -> list[tuple[int, int, float]]:
    """The first `limit` 2Q gates from program[start:] as (a, b, weight),
    weights 1, decay, decay**2, ... so nearer gates count more."""
    pairs = []
    weight = 1.0
    for op in program[start:]:
        if len(pairs) == limit:
            break
        if op[0] == "2Q":
            pairs.append((op[1], op[2], weight))
            weight *= decay
    return pairs


def partner_counts(program: list[tuple]) -> dict[int, Counter]:
    """partner_counts(program)[a][b] = how many 2Q gates act on the pair a, b."""
    counts: dict[int, Counter] = defaultdict(Counter)
    for op in program:
        if op[0] == "2Q":
            counts[op[1]][op[2]] += 1
            counts[op[2]][op[1]] += 1
    return counts
