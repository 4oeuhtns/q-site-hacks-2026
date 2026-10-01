import copy

from starter_kit import BENCHMARKS, baseline_solve, core_score, score_summary

from period_router import solve
from period_router.random_programs import STYLES, random_program, unseen_chips
from solver.cost import cost


def test_team_cost_matches_the_official_score(hw):
    # solve() ranks candidates with the team's solver.cost; it must agree with the grader.
    for program in BENCHMARKS.values():
        _, routed = baseline_solve(program, hw)
        assert cost(routed) == core_score(routed)


def test_every_benchmark_is_valid_and_the_total_beats_the_baseline(hw):
    scores = {}
    for name, program in BENCHMARKS.items():
        placement, routed = solve(program, hw)
        result = score_summary(program, hw, placement, routed)
        assert result["valid"], (name, result["message"])
        scores[name] = result["score"]
    assert scores["chain_trotter"] == 4.5  # zero SWAPs: the best possible
    assert scores["vqe_layers"] == 3.0  # zero SWAPs: the best possible
    assert sum(scores.values()) < 100  # baseline: 283.5


def test_inputs_are_not_modified(hw):
    program = BENCHMARKS["dense_random"]
    program_before = copy.deepcopy(program)
    edges_before = sorted(hw.edges)
    solve(program, hw)
    assert program == program_before
    assert sorted(hw.edges) == edges_before


def test_random_programs_on_chips_it_was_never_tuned_on(hw):
    chips = {"challenge": hw, **unseen_chips()}
    for chip_name, graph in chips.items():
        for style in STYLES:
            for seed in range(3):
                program = random_program(6 + 4 * seed, 12 + 10 * seed, style, seed)
                placement, routed = solve(program, graph)
                result = score_summary(program, graph, placement, routed)
                assert result["valid"], (chip_name, style, seed, result["message"])
                baseline = score_summary(program, graph, *baseline_solve(program, graph))
                assert result["score"] <= baseline["score"], (chip_name, style, seed)
