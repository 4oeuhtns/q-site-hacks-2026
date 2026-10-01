from starter_kit import BENCHMARKS, score_summary

from period_router.initial import initial_placement
from period_router.periods import route_periods


def test_worked_example(hw, chip):
    # A line 0-1-2-3, then 3 talks to 0 and 0 talks to 2.
    program = [("2Q", 0, 1), ("2Q", 1, 2), ("2Q", 2, 3), ("2Q", 3, 0), ("2Q", 0, 2)]
    placement = {0: 5, 1: 6, 2: 7, 3: 11}
    stats = {}
    routed = route_periods(program, placement, chip, stats)
    result = score_summary(program, hw, placement, routed)
    assert result["valid"]
    assert (result["swap_count"], result["depth"], result["score"]) == (2, 6, 5.0)
    assert [length for length, _ in stats["periods"]] == [3, 2]


def test_valid_on_every_benchmark(hw, chip):
    for name, program in BENCHMARKS.items():
        placement = initial_placement(program, chip)
        routed = route_periods(program, placement, chip)
        result = score_summary(program, hw, placement, routed)
        assert result["valid"], (name, result["message"])


def test_programs_that_fit_need_no_swaps(hw, chip):
    for name in ("chain_trotter", "vqe_layers"):
        program = BENCHMARKS[name]
        placement = initial_placement(program, chip)
        routed = route_periods(program, placement, chip)
        assert not any(op[0] == "SWAP" for op in routed), name
