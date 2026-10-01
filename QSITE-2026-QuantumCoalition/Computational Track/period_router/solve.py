"""solve(): try every router from every starting seating, keep the best valid result.

Starting seatings:
  1. initial_placement      - first period near the centre, the rest by partners
  2. label_order_placement  - qubit k on chair k (good when labels follow the wiring)
Each is then refined with the forward-backward trick (from SABRE): route the
program forwards, then backwards from where everyone ended up; where they end up
after that is a seating shaped by the whole program. Refining continues while it
lowers the score, up to MAX_REFINE_ROUNDS.
"""

import networkx as nx

from solver.cost import cost, validate

from .chip import Chip, make_chip
from .initial import initial_placement, label_order_placement
from .layout import Layout
from .lookahead import route_lookahead
from .periods import route_periods

ROUTERS = (route_periods, route_lookahead)
MAX_REFINE_ROUNDS = 5


def final_placement(placement: dict[int, int], routed: list[tuple]) -> dict[int, int]:
    """Where every logical qubit sits after all the SWAPs in `routed`."""
    layout = Layout(placement)
    for op in routed:
        if op[0] == "SWAP":
            layout.swap(op[1], op[2])
    return layout.where


def forward_backward(program: list[tuple], placement: dict[int, int], chip: Chip) -> dict[int, int]:
    after_forward = final_placement(placement, route_periods(program, placement, chip))
    return final_placement(after_forward, route_periods(program[::-1], after_forward, chip))


def solve(program: list[tuple], hardware_graph: nx.Graph) -> tuple[dict[int, int], list[tuple]]:
    """
    Args:
        program: list of tuples
            ("2Q", i, j) = two-qubit gate between logical qubits i and j
            ("1Q", i)    = single-qubit gate on logical qubit i
        hardware_graph: networkx.Graph
            Nodes are physical qubit indices, edges are connections

    Returns:
        initial_placement: dict
            Maps logical qubit index → physical qubit index
            e.g. {0: 5, 1: 6, 2: 9, 3: 10}
        routed_program: list of tuples
            Same format as input, but on PHYSICAL qubits,
            with ("SWAP", p, q) operations inserted as needed.
            Every ("2Q", p, q) must have (p,q) as an edge in hardware_graph.
    """
    chip = make_chip(hardware_graph)
    best: tuple[dict[int, int], list[tuple]] | None = None

    def best_from(placement: dict[int, int]) -> float:
        """Route from `placement` with every router; keep the overall best. Returns this seed's best cost."""
        nonlocal best
        seed_best = float("inf")
        for router in ROUTERS:
            routed = router(program, placement, chip)
            if not validate(program, hardware_graph, placement, routed):
                raise RuntimeError(f"{router.__name__} produced an invalid program")  # a bug
            score = cost(routed)
            seed_best = min(seed_best, score)
            if best is None or score < cost(best[1]):
                best = (dict(placement), routed)
        return seed_best

    for placement in (initial_placement(program, chip), label_order_placement(program, chip)):
        score = best_from(placement)
        for _ in range(MAX_REFINE_ROUNDS):
            placement = forward_backward(program, placement, chip)
            refined = best_from(placement)
            if refined >= score:
                break
            score = refined
    return best
