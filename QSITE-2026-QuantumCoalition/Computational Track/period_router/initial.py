"""Starting seating: the first period seated near the chip's centre, then
everyone else placed next to the partners they talk to most."""

from .chip import Chip
from .fit import longest_fitting_end
from .lookahead import DECAY, LOOKAHEAD
from .program import first_appearance, interaction_graph, logical_qubits, partner_counts, upcoming_pairs
from .seating import closest_seating


def initial_placement(program: list[tuple], chip: Chip) -> dict[int, int]:
    qubits = logical_qubits(program)
    if len(qubits) > chip.graph.number_of_nodes():
        raise ValueError(f"program uses {len(qubits)} qubits but the chip has {chip.graph.number_of_nodes()}")
    if not program:
        return {}

    first_end = longest_fitting_end(program, 0, chip)
    first_period = interaction_graph(program[:first_end])
    upcoming = upcoming_pairs(program, first_end, LOOKAHEAD, DECAY)
    placement = closest_seating(first_period, {q: chip.center for q in first_period}, chip, upcoming)

    partners = partner_counts(program)
    dist = chip.dist
    for qubit in first_appearance(program):
        if qubit in placement:
            continue
        taken = set(placement.values())
        free = [p for p in chip.graph.nodes if p not in taken]

        def pull(chair: int) -> tuple:
            to_partners = sum(
                count * dist[chair][placement[partner]]
                for partner, count in partners[qubit].items()
                if partner in placement
            )
            return to_partners, dist[chip.center][chair], chair

        placement[qubit] = min(free, key=pull)
    return placement


def label_order_placement(program: list[tuple], chip: Chip) -> dict[int, int]:
    """Logical qubits in label order onto physical qubits in label order.

    Naive, but when a program's labels follow the chip's wiring (e.g. qubit k
    talks to k+1 on a ring) it is already near-perfect, so it is worth trying.
    """
    return dict(zip(logical_qubits(program), sorted(chip.graph.nodes)))
