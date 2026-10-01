"""Print how the period router compares with the baseline.

Run from the Computational Track folder:
    python -m uv run python -m period_router.compare
"""

import statistics
import time

from starter_kit import BENCHMARKS, baseline_solve, build_hardware_graph, score_summary

from . import solve
from .random_programs import STYLES, random_program, unseen_chips


def checked_score(program, graph, solver) -> dict:
    placement, routed = solver(program, graph)
    result = score_summary(program, graph, placement, routed)
    if not result["valid"]:
        raise AssertionError(f"{solver.__name__}: {result['message']}")
    return result


def benchmark_table() -> None:
    graph = build_hardware_graph()
    print(f"{'benchmark':16s}{'baseline':>10s}{'ours':>8s}{'swaps':>7s}{'depth':>7s}")
    base_total = ours_total = 0.0
    for name, program in BENCHMARKS.items():
        base = checked_score(program, graph, baseline_solve)
        ours = checked_score(program, graph, solve)
        base_total += base["score"]
        ours_total += ours["score"]
        print(f"{name:16s}{base['score']:10.1f}{ours['score']:8.1f}{ours['swap_count']:7d}{ours['depth']:7d}")
    print(f"{'TOTAL':16s}{base_total:10.1f}{ours_total:8.1f}")


def random_table(seeds: int = 10) -> None:
    chips = {"challenge": build_hardware_graph(), **unseen_chips()}
    print(f"{'chip':14s}{'programs':>9s}{'mean ours/baseline':>20s}{'worst':>7s}")
    for chip_name, graph in chips.items():
        ratios = []
        for style in STYLES:
            for seed in range(seeds):
                program = random_program(6 + (seed % 6) * 2, 10 + seed * 6, style, seed)
                ours = checked_score(program, graph, solve)["score"]
                base = checked_score(program, graph, baseline_solve)["score"]
                ratios.append(ours / base)
        print(f"{chip_name:14s}{len(ratios):9d}{statistics.mean(ratios):20.3f}{max(ratios):7.3f}")


if __name__ == "__main__":
    start = time.time()
    benchmark_table()
    print()
    random_table()
    print(f"\n({time.time() - start:.0f}s)")
