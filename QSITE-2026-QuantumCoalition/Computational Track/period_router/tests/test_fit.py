import networkx as nx

from starter_kit import BENCHMARKS

from period_router.fit import cannot_fit, fits, longest_fitting_end


def test_a_line_of_20_fits(chip):
    assert fits(nx.path_graph(20), chip)


def test_a_hexagon_fits(chip):
    assert fits(nx.cycle_graph(6), chip)


def test_a_square_is_ruled_out_instantly(chip):
    assert cannot_fit(nx.cycle_graph(4), chip)  # shorter than the chip's shortest loop


def test_a_triangle_is_ruled_out_instantly(chip):
    assert cannot_fit(nx.cycle_graph(3), chip)  # odd loop on a bipartite chip


def test_four_partners_is_ruled_out_instantly(chip):
    assert cannot_fit(nx.star_graph(4), chip)  # centre has 4 partners, chairs have at most 3


def test_search_rejects_a_tree_the_quick_checks_miss(chip):
    tree = nx.balanced_tree(2, 3)  # 15 nodes; networkx's exact VF2 also says it cannot fit
    assert not cannot_fit(tree, chip)
    assert not fits(tree, chip)


def test_chain_trotter_is_one_period(chip):
    program = BENCHMARKS["chain_trotter"]
    assert longest_fitting_end(program, 0, chip) == len(program)


def test_period_ends_before_a_square_closes(chip):
    program = [("2Q", 0, 1), ("2Q", 1, 2), ("2Q", 2, 3), ("2Q", 3, 0), ("2Q", 0, 2)]
    assert longest_fitting_end(program, 0, chip) == 3
    assert longest_fitting_end(program, 3, chip) == 5
