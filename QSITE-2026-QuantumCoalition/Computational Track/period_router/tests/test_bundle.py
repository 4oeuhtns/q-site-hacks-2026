import json
import subprocess
import sys

from starter_kit import BENCHMARKS

from period_router import solve
from period_router.bundle import build_source
from period_router.random_programs import random_program

# Runs inside a fresh Python that can only see the bundled file (plus installed
# packages): no period_router, no solver, no starter_kit.
STANDALONE = """
import json, sys
sys.path.insert(0, ".")  # the temp folder, which holds only submission.py
import networkx as nx
for blocked in ("period_router", "solver", "starter_kit"):
    try:
        __import__(blocked)
        raise SystemExit(f"{blocked} should not be importable here")
    except ImportError:
        pass
from submission import solve
data = json.load(sys.stdin)
graph = nx.Graph([tuple(e) for e in data["edges"]])
results = []
for program in data["programs"]:
    placement, routed = solve([tuple(op) for op in program], graph)
    results.append([sorted(placement.items()), [list(op) for op in routed]])
json.dump(results, sys.stdout)
"""


def test_bundle_is_self_contained_and_matches_the_package(hw, tmp_path):
    (tmp_path / "submission.py").write_text(build_source(), encoding="utf-8")
    programs = list(BENCHMARKS.values()) + [random_program(12, 40, style, 3) for style in ("random", "local")]

    run = subprocess.run(
        [sys.executable, "-I", "-c", STANDALONE],
        input=json.dumps({"edges": list(hw.edges), "programs": programs}),
        capture_output=True, text=True, cwd=tmp_path,
    )
    assert run.returncode == 0, run.stderr

    expected = []
    for program in programs:
        placement, routed = solve(program, hw)
        expected.append([sorted(placement.items()), [list(op) for op in routed]])
    assert json.loads(run.stdout) == json.loads(json.dumps(expected))


def test_bundle_solve_has_the_spec_docstring():
    namespace = {}
    exec(compile(build_source(), "submission.py", "exec"), namespace)
    doc = namespace["solve"].__doc__
    assert "initial_placement: dict" in doc
    assert 'Every ("2Q", p, q) must have (p,q) as an edge in hardware_graph.' in doc
