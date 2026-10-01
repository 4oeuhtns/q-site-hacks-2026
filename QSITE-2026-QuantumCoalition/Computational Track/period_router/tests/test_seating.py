import networkx as nx

from period_router.seating import closest_seating, search_seating


def assert_is_seating(pattern, seat, chip):
    assert set(seat) == set(pattern.nodes)
    assert len(set(seat.values())) == len(seat), "two qubits on one chair"
    for a, b in pattern.edges:
        assert seat[b] in chip.neighbors[seat[a]], f"{a}-{b} not on neighbouring chairs"


def test_search_finds_a_seating_for_a_long_line(chip):
    pattern = nx.path_graph(16)
    assert_is_seating(pattern, search_seating(pattern, chip), chip)


def test_search_returns_none_when_there_is_no_seating(chip):
    assert search_seating(nx.balanced_tree(2, 3), chip) is None


def test_closest_seating_keeps_a_seating_that_already_works(chip):
    current = {0: 5, 1: 6, 2: 7}
    pattern = nx.Graph([(0, 1), (1, 2)])
    assert closest_seating(pattern, current, chip) == current


def test_closest_seating_moves_as_little_as_possible(chip):
    # L0 must end up between L3 and L2. The cheapest way moves 3 hops in total.
    current = {0: 5, 1: 6, 2: 7, 3: 11}
    pattern = nx.Graph([(3, 0), (0, 2)])
    seat = closest_seating(pattern, current, chip)
    assert_is_seating(pattern, seat, chip)
    assert sum(chip.dist[current[q]][seat[q]] for q in seat) == 3


def test_upcoming_gates_change_the_choice(chip):
    # Seat the pair 0-1 around chair 9; chairs 5, 8 and 10 are equally close.
    # L2 sits on chair 12 and gate (1, 2) comes next, so the look-ahead should
    # pick chair 8 for L1: it is next to chair 12.
    pattern = nx.Graph([(0, 1)])
    reference = {0: 9, 1: 9, 2: 12}
    assert closest_seating(pattern, reference, chip) == {0: 9, 1: 5}
    assert closest_seating(pattern, reference, chip, upcoming=[(1, 2, 1.0)]) == {0: 9, 1: 8}
