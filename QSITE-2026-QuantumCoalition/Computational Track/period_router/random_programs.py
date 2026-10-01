"""Random programs and chips for testing that the router works on any input,
not just the six benchmarks."""

import random

import networkx as nx

STYLES = ("random", "local", "clustered")


def random_program(n_qubits: int, n_gates: int, style: str, seed: int) -> list[tuple]:
    """A program of `n_gates` 2Q gates on labels 0..n_qubits-1, plus some 1Q gates.

    random    - any two qubits
    local     - qubits whose labels differ by 1 or 2 (structured, like chains)
    clustered - mostly within groups of 4, occasionally across groups
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    rng = random.Random(seed)
    program: list[tuple] = []
    for _ in range(n_gates):
        if style == "random":
            a, b = rng.sample(range(n_qubits), 2)
        elif style == "local":
            a = rng.randrange(n_qubits)
            b = min(max(a + rng.choice((-2, -1, 1, 2)), 0), n_qubits - 1)
            if a == b:
                b = a + 1 if a + 1 < n_qubits else a - 1
        else:
            if rng.random() < 0.8:
                start = 4 * rng.randrange((n_qubits + 3) // 4)
                group = [q for q in range(start, start + 4) if q < n_qubits]
                if len(group) < 2:
                    group = [n_qubits - 2, n_qubits - 1]
                a, b = rng.sample(group, 2)
            else:
                a, b = rng.sample(range(n_qubits), 2)
        program.append(("2Q", a, b))
        if rng.random() < 0.2:
            program.append(("1Q", rng.randrange(n_qubits)))
    return program


def unseen_chips() -> dict[str, nx.Graph]:
    """Chips the router was never tuned on. All connected, all with 20 qubits."""
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 5))
    ring = nx.cycle_graph(20)
    regular = nx.random_regular_graph(3, 20, seed=1)
    return {"grid_4x5": grid, "ring_20": ring, "regular3_20": regular}
