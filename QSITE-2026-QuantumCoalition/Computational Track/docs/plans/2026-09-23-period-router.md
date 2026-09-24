# Period Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `period_router`, a from-scratch `solve(program, hardware_graph)` for the Computational Track that scores **89.0** on the six benchmarks (baseline: 283.5), and stays valid and at or below the baseline's score on random programs and on chips it was never tuned on.

**Architecture:** Cut the program into *periods*: the longest stretches of gates that can all run with zero SWAPs under a single seating (a subgraph monomorphism, found by our own step-budgeted search). For each period, either reshuffle (token swapping) to the closest seating where the whole period fits, choosing seatings that also look ahead to the next gates, or route the period gate by gate with a lookahead router, whichever scores better. `solve()` runs both routers from two starting seatings, refines each seating with SABRE's forward-backward trick for as long as it helps, and returns the best valid result.

**Tech Stack:** Python 3.14, networkx 3.7, pytest (fetched per run with `uv --with`, not added to the project), the organisers' `starter_kit/`, the team's `solver/cost.py`.

---

## Before you start

**Where everything happens.** Every path in this plan is relative to the track folder, `QSITE-2026-QuantumCoalition/Computational Track/`. Run every command from there:

```bash
cd "QSITE-2026-QuantumCoalition/Computational Track"
```

**Branch.** `period-router`, created from `main`. Commit after every task. Do not push; that is the team's call.

**Environment.** `.venv/` in the track folder, made by `python -m uv sync` (Python 3.14.7, networkx 3.7, pennylane 0.45.1). On this machine `uv` is not on PATH, so always type `python -m uv`. The first import of pennylane may print a `SyntaxWarning` from the `autograd` package. It is harmless; ignore it.

**Running tests.** Every test command has this shape:

```bash
python -m uv run --with pytest pytest period_router/tests/test_chip.py -v
```

`--with pytest` brings pytest in for that run only; `pyproject.toml` and `uv.lock` are never touched. `period_router/tests/pytest.ini` (Task 1) switches off pytest's cache folder, which `.gitignore` doesn't cover, and silences the `autograd` warning.

**What this plan does not touch.** `plan.md`, `solver/cost.py`, `starter_kit/`, and everything outside the track folder. The code lives in its own package, `period_router/`, so it cannot collide with the modules `plan.md` proposes (`solver/router.py`, `solver/embed.py`, …). It *uses* `solver/cost.py` (`cost`, `validate`). Those were checked against the official scorer on every baseline answer and agree exactly, and Task 11 adds a test that keeps it that way.

**Rules the scorer enforces silently** (any break scores ∞ for that benchmark):
1. Real gates must come out in exactly the program's order. Only SWAPs may be inserted.
2. Operand order is kept: `("2Q", a, b)` becomes `("2Q", where[a], where[b])`, never sorted.
3. The placement lists exactly the logical qubits the program uses. `qaoa_random` never uses 4 or 7, and placing them is an error.
4. The returned placement is the *starting* seating, not where qubits ended up.

**Vocabulary.**

| Word | Meaning |
|---|---|
| logical qubit | a qubit named in the program ("student") |
| physical qubit / chair | a node of the hardware graph |
| seating, placement | dict logical → physical |
| fits | a stretch of gates can run with zero SWAPs under one seating: its interaction graph is subgraph-monomorphic to the chip |
| period | the longest stretch from the current gate that fits |
| reshuffle | SWAPs that move some qubits onto target chairs (the *token swapping* problem) |

## Results this plan reproduces

Measured by building exactly the code in this plan (Task 12 prints these):

```
benchmark         baseline    ours  swaps  depth
ghz_star              14.0     8.0      3     10
chain_trotter         15.0     4.5      0      9
ladder_trotter        35.5     9.0      5      8
qaoa_random           39.0    15.0      9     12
dense_random         122.0    49.5     34     31
vqe_layers            58.0     3.0      0      6
TOTAL                283.5    89.0

chip           programs  mean ours/baseline  worst
challenge            30               0.446  0.622
grid_4x5             30               0.321  0.526
ring_20              30               0.462  0.656
regular3_20          30               0.293  0.532
```

`chain_trotter` and `vqe_layers` are optimal (zero SWAPs). Every setting was chosen on random programs, not on the benchmarks. Appendix A lists each measurement.

## File structure

| File | Responsibility |
|---|---|
| `period_router/__init__.py` | Exports `solve` |
| `period_router/chip.py` | Facts about the hardware graph: distances, neighbours, max degree, girth, bipartite, centre |
| `period_router/program.py` | Reading a program: used qubits, interaction graph, upcoming gates, partner counts |
| `period_router/layout.py` | `Layout`: who sits where, in both directions; `swap`, `emit` |
| `period_router/random_programs.py` | Random programs and unseen chips for testing generality |
| `period_router/seating.py` | Budgeted backtracking search for seatings; `closest_seating` with look-ahead |
| `period_router/fit.py` | Instant "can't fit" proofs, `fits`, binary search for the period's end |
| `period_router/token_swap.py` | Reshuffle onto target chairs: greedy, then a spanning-tree fallback that always finishes |
| `period_router/lookahead.py` | Gate-by-gate router with lookahead (used for periods and as a standalone router) |
| `period_router/initial.py` | Two starting seatings: centre-out, and label order |
| `period_router/periods.py` | The period router: per period, reseat or route, whichever scores better |
| `period_router/solve.py` | Portfolio: seeds × routers, forward-backward refinement, validation |
| `period_router/compare.py` | Prints the results tables for the writeup |
| `period_router/tests/…` | `pytest.ini`, `conftest.py`, one test file per module |
| `starter.ipynb` (modify) | Cell 20 calls `period_router.solve` on all six benchmarks |

Dependency order (each task only needs the ones before it): chip → program → layout → random_programs → seating → fit → token_swap → lookahead → initial → periods → solve.

---

### Task 0: Preflight

- [ ] **Step 1: Commit this plan, so the branch history starts with it**

```bash
git add docs/plans/2026-09-23-period-router.md
git commit -m "docs: period router implementation plan"
```

- [ ] **Step 2: Confirm the branch**

Run: `git status -sb`
Expected: the first line is `## period-router`. The only other line may be `?? uv.lock`, created by environment setup. Leave it; Task 14 decides what to do with it.

- [ ] **Step 3: Confirm the environment reproduces the baseline**

Run:
```bash
python -m uv run python -c "from starter_kit import BENCHMARKS, build_hardware_graph, baseline_solve, core_score; g = build_hardware_graph(); print(sum(core_score(baseline_solve(p, g)[1]) for p in BENCHMARKS.values()))"
```
Expected: `283.5` (after the harmless warning, if it appears). If the number differs, stop: every later measurement depends on it.

---

### Task 1: Package skeleton and chip facts

Everything the router knows about the hardware is computed from the graph it is given. Nothing is hard-coded to the 20-qubit chip, which is what lets the same code run on the grid, ring and random chips in later tests.

**Files:**
- Create: `period_router/__init__.py`, `period_router/tests/pytest.ini`, `period_router/tests/conftest.py`
- Create: `period_router/chip.py`
- Test: `period_router/tests/test_chip.py`

- [ ] **Step 1: Create the package marker**

`period_router/__init__.py` (Task 11 replaces this; exporting `solve` now would break every import until `solve.py` exists):

```python
"""Period router for the QSITE 2026 Computational Track."""
```

- [ ] **Step 2: Create the pytest config**

`period_router/tests/pytest.ini`:

```ini
[pytest]
addopts = -p no:cacheprovider
filterwarnings =
    ignore::SyntaxWarning
```

- [ ] **Step 3: Create the shared fixtures**

`period_router/tests/conftest.py`:

```python
import sys
from pathlib import Path

import pytest

# Make `starter_kit`, `solver` and `period_router` importable: they all live in
# the Computational Track folder, two levels above this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starter_kit import build_hardware_graph  # noqa: E402

from period_router.chip import make_chip  # noqa: E402


@pytest.fixture
def hw():
    """The challenge's 20-qubit hardware graph."""
    return build_hardware_graph()


@pytest.fixture
def chip(hw):
    return make_chip(hw)
```

- [ ] **Step 4: Write the failing test**

`period_router/tests/test_chip.py`:

```python
import networkx as nx
import pytest

from period_router.chip import make_chip


def test_challenge_chip_facts(chip):
    assert chip.max_degree == 3
    assert chip.girth == 6
    assert chip.bipartite is True
    assert chip.center == 9
    assert chip.dist[0][19] == 7
    assert chip.neighbors[5] == frozenset({4, 6, 9})


def test_facts_come_from_the_graph_not_the_challenge():
    ring = make_chip(nx.cycle_graph(7))
    assert ring.max_degree == 2
    assert ring.girth == 7
    assert ring.bipartite is False  # odd ring


def test_disconnected_graph_is_rejected():
    with pytest.raises(ValueError):
        make_chip(nx.Graph([(0, 1), (2, 3)]))
```

- [ ] **Step 5: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_chip.py -v`
Expected: FAIL. pytest reports an error loading `conftest.py` containing `No module named 'period_router.chip'`.

- [ ] **Step 6: Implement**

`period_router/chip.py`:

```python
"""Facts about the hardware graph, computed once per solve() call.

Everything the router needs to know about the chip is derived here from the
graph it is given, so nothing in the package is specific to the 20-qubit chip.
"""

from dataclasses import dataclass

import networkx as nx


@dataclass
class Chip:
    graph: nx.Graph
    dist: dict  # dist[p][q] = number of hops between physical qubits p and q
    max_degree: int  # most neighbours any physical qubit has
    girth: float  # length of the shortest loop (inf if the graph has none)
    bipartite: bool  # True if every loop has even length
    center: int  # physical qubit closest to everything; used to seed placement
    neighbors: dict  # neighbors[p] = frozenset of p's neighbours (plain dict: fast in hot loops)


def make_chip(graph: nx.Graph) -> Chip:
    if not nx.is_connected(graph):
        raise ValueError("hardware graph must be connected")
    dist = dict(nx.all_pairs_shortest_path_length(graph))
    return Chip(
        graph=graph,
        neighbors={p: frozenset(graph.neighbors(p)) for p in graph.nodes},
        dist=dist,
        max_degree=max(degree for _, degree in graph.degree()),
        girth=nx.girth(graph),
        bipartite=nx.is_bipartite(graph),
        center=min(graph.nodes, key=lambda p: (max(dist[p].values()), -graph.degree(p), p)),
    )
```

- [ ] **Step 7: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_chip.py -v`
Expected: `3 passed`

- [ ] **Step 8: Commit**

```bash
git add period_router/__init__.py period_router/chip.py period_router/tests/pytest.ini period_router/tests/conftest.py period_router/tests/test_chip.py
git commit -m "period router: chip facts derived from the hardware graph"
```

---

### Task 2: Reading a program

**Files:**
- Create: `period_router/program.py`
- Test: `period_router/tests/test_program.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_program.py`:

```python
from collections import Counter

from period_router.program import (
    first_appearance,
    interaction_graph,
    logical_qubits,
    partner_counts,
    upcoming_pairs,
)

PROGRAM = [("2Q", 0, 1), ("1Q", 5), ("2Q", 1, 2), ("2Q", 0, 1)]


def test_logical_qubits_allows_gaps_in_labels():
    assert logical_qubits([("2Q", 3, 7), ("1Q", 9)]) == [3, 7, 9]


def test_first_appearance():
    assert first_appearance(PROGRAM) == [0, 1, 5, 2]


def test_interaction_graph_ignores_1q_gates_and_repeats():
    graph = interaction_graph(PROGRAM)
    assert sorted(graph.edges) == [(0, 1), (1, 2)]
    assert 5 not in graph


def test_upcoming_pairs_weights_decay():
    assert upcoming_pairs(PROGRAM, 1, limit=2, decay=0.5) == [(1, 2, 1.0), (0, 1, 0.5)]
    assert upcoming_pairs(PROGRAM, 4, limit=5, decay=0.5) == []


def test_partner_counts():
    counts = partner_counts(PROGRAM)
    assert counts[0] == Counter({1: 2})
    assert counts[1] == Counter({0: 2, 2: 1})
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_program.py -v`
Expected: FAIL with `No module named 'period_router.program'`

- [ ] **Step 3: Implement**

`period_router/program.py`:

```python
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
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_program.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/program.py period_router/tests/test_program.py
git commit -m "period router: program helpers (qubits, interaction graph, upcoming gates)"
```

---

### Task 3: Layout, the running record of who sits where

Every router mutates a `Layout` as it inserts SWAPs, and uses `emit` to write each gate on physical qubits. Keeping both directions (logical → physical and physical → logical) avoids the "which one did I update" bug. `emit` keeps operand order (rule 2 above).

**Files:**
- Create: `period_router/layout.py`
- Test: `period_router/tests/test_layout.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_layout.py`:

```python
import pytest

from period_router.layout import Layout


def test_swap_exchanges_two_qubits():
    layout = Layout({0: 5, 1: 6})
    layout.swap(5, 6)
    assert layout.where == {0: 6, 1: 5}
    assert layout.occupant == {6: 0, 5: 1}


def test_swap_into_an_empty_chair():
    layout = Layout({0: 5})
    layout.swap(5, 9)
    assert layout.where == {0: 9}
    assert layout.occupant == {9: 0}


def test_copy_is_independent():
    layout = Layout({0: 5, 1: 6})
    copy = layout.copy()
    copy.swap(5, 6)
    assert layout.where == {0: 5, 1: 6}


def test_emit_keeps_operand_order():
    layout = Layout({0: 5, 1: 6})
    assert layout.emit(("2Q", 1, 0)) == ("2Q", 6, 5)
    assert layout.emit(("1Q", 1)) == ("1Q", 6)
    with pytest.raises(ValueError):
        layout.emit(("3Q", 0, 1, 2))
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_layout.py -v`
Expected: FAIL with `No module named 'period_router.layout'`

- [ ] **Step 3: Implement**

`period_router/layout.py`:

```python
"""Who sits where, kept in both directions and updated on every SWAP."""


class Layout:
    def __init__(self, placement: dict[int, int]):
        self.where = dict(placement)  # logical -> physical
        self.occupant = {physical: logical for logical, physical in placement.items()}  # physical -> logical

    def copy(self) -> "Layout":
        return Layout(self.where)

    def swap(self, p: int, q: int) -> None:
        """Exchange whatever sits on physical qubits p and q (either may be empty)."""
        a = self.occupant.pop(p, None)
        b = self.occupant.pop(q, None)
        if a is not None:
            self.occupant[q] = a
            self.where[a] = q
        if b is not None:
            self.occupant[p] = b
            self.where[b] = p

    def emit(self, op: tuple) -> tuple:
        """Rewrite a program op onto physical qubits, keeping the operand order."""
        if op[0] == "2Q":
            return ("2Q", self.where[op[1]], self.where[op[2]])
        if op[0] == "1Q":
            return ("1Q", self.where[op[1]])
        raise ValueError(f"unknown op kind: {op[0]}")
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_layout.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/layout.py period_router/tests/test_layout.py
git commit -m "period router: Layout tracks seating in both directions"
```

---

### Task 4: Random programs and unseen chips

Built early because later tests use them to prove the router works on inputs it was never tuned on. The helper is named `unseen_chips`, not `test_chips`: pytest would try to run any imported function whose name starts with `test_`.

**Files:**
- Create: `period_router/random_programs.py`
- Test: `period_router/tests/test_random_programs.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_random_programs.py`:

```python
import networkx as nx
import pytest

from period_router.random_programs import STYLES, random_program, unseen_chips


@pytest.mark.parametrize("style", STYLES)
def test_program_shape(style):
    program = random_program(8, 30, style, seed=1)
    two_qubit = [op for op in program if op[0] == "2Q"]
    assert len(two_qubit) == 30
    assert all(op[1] != op[2] for op in two_qubit)
    assert all(0 <= q < 8 for op in program for q in op[1:])
    assert program == random_program(8, 30, style, seed=1)  # same seed, same program


def test_unknown_style_is_an_error():
    with pytest.raises(ValueError):
        random_program(8, 30, "sideways", seed=1)


def test_unseen_chips_are_connected_and_20_qubits():
    for name, graph in unseen_chips().items():
        assert graph.number_of_nodes() == 20, name
        assert nx.is_connected(graph), name
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_random_programs.py -v`
Expected: FAIL with `No module named 'period_router.random_programs'`

- [ ] **Step 3: Implement**

`period_router/random_programs.py`:

```python
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
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_random_programs.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/random_programs.py period_router/tests/test_random_programs.py
git commit -m "period router: random programs and unseen chips for generality tests"
```

---

### Task 5: Seating search

The heart of the method. A backtracking search that seats the period's qubits in breadth-first order. Every qubit after the first in its group then has a seated partner, so it has at most 3 candidate chairs. It is used two ways: find *any* seating (the fit check, Task 6), and find the *cheapest* one (branch and bound, trying nearest chairs first).

Why not networkx's `GraphMatcher` (VF2)? Measured: proving that one 16-qubit pattern did *not* fit a 4×5 grid took VF2 **48 seconds**, and VF2 can't be interrupted. This search has a step budget. When the budget runs out it answers "no seating", which is always safe: the period just ends earlier.

The cost of a seating is (hops moved from the current seats) + `UPCOMING_WEIGHT` × (how far apart the next 20 gates' pairs would end up). The second term is the look-ahead across periods, and it is what took `ladder_trotter` from 19.5 to 9.0.

**Files:**
- Create: `period_router/seating.py`
- Test: `period_router/tests/test_seating.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_seating.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_seating.py -v`
Expected: FAIL with `No module named 'period_router.seating'`

- [ ] **Step 3: Implement**

`period_router/seating.py`:

```python
"""Seatings: where to put a period's qubits so every interacting pair are neighbours.

In graph terms a seating is a subgraph monomorphism from the period's
interaction graph into the chip. We find them with our own backtracking search
rather than networkx's VF2, because ours has a step budget: VF2 can take
exponential time to prove that a large pattern does NOT fit, and cannot be
interrupted.

When choosing among seatings, the cost of a seating is
    (total hops the period's qubits move from where they sit now)
  + UPCOMING_WEIGHT x (how far apart the next gates' pairs would end up)
The second term is the look-ahead across periods: prefer seatings that are
cheap to leave, not just cheap to reach.
"""

import math
from collections import defaultdict

import networkx as nx

from .chip import Chip

FIT_BUDGET = 20_000  # steps to spend proving a seating exists
CLOSEST_BUDGET = 2_000  # steps to spend improving on the first seating found
UPCOMING_WEIGHT = 1.0  # one hop of future distance counts the same as one hop moved now


def search_order(pattern: nx.Graph) -> list[int]:
    """Breadth-first order, so every qubit after the first in its group has a
    partner already seated. That caps its options at that chair's neighbours."""
    order = []
    components = sorted(nx.connected_components(pattern), key=lambda c: (-len(c), min(c)))
    for component in components:
        root = max(sorted(component), key=pattern.degree)
        order.append(root)
        order.extend(child for _, child in nx.bfs_edges(pattern, root))
    return order


def search_seating(
    pattern: nx.Graph,
    chip: Chip,
    reference: dict[int, int] | None = None,
    upcoming: list[tuple[int, int, float]] = (),
    budget: int = FIT_BUDGET,
) -> dict[int, int] | None:
    """Find a seating (logical -> physical) for every node of `pattern`.

    reference=None: return the first seating found.
    reference given: return the cheapest seating found (see module docstring).
    upcoming: (a, b, weight) pairs for the gates after this period.
    Returns None if the budget runs out before any seating is found.
    """
    order = search_order(pattern)
    dist, neighbors = chip.dist, chip.neighbors
    partners = {node: list(pattern.neighbors(node)) for node in order}
    need = {node: len(partners[node]) for node in order}
    all_chairs = frozenset(neighbors)
    seat: dict[int, int] = {}
    used: set[int] = set()
    best_cost = math.inf
    best_seat: dict[int, int] | None = None
    steps = 0

    pairs_of = defaultdict(list)  # node -> [(other qubit, weight)]
    for a, b, weight in upcoming:
        if a in pattern:
            pairs_of[a].append((b, weight))
        if b in pattern:
            pairs_of[b].append((a, weight))

    def added_cost(node: int, chair: int) -> float:
        """Cost of seating `node` on `chair`, given who is seated already."""
        if reference is None:
            return 0.0
        total = float(dist[reference[node]][chair])
        for other, weight in pairs_of[node]:
            if other in seat:  # both in the period: count once, when the second is seated
                total += UPCOMING_WEIGHT * weight * (dist[chair][seat[other]] - 1)
            elif other not in pattern and other in reference:  # other stays where it is
                total += UPCOMING_WEIGHT * weight * max(0, dist[chair][reference[other]] - 1)
        return total

    def candidates(node: int) -> list[tuple[float, int]]:
        seated_partners = [seat[n] for n in partners[node] if n in seat]
        if seated_partners:
            options = neighbors[seated_partners[0]]
            for chair in seated_partners[1:]:
                options = options & neighbors[chair]
        else:
            options = all_chairs
        scored = [(added_cost(node, c), c) for c in options if c not in used and len(neighbors[c]) >= need[node]]
        return sorted(scored)

    def extend(k: int, cost: float) -> bool:
        """Returns True when the search should stop."""
        nonlocal best_cost, best_seat, steps
        if k == len(order):
            best_cost, best_seat = cost, dict(seat)
            return reference is None or cost == 0  # nothing can beat zero
        node = order[k]
        for extra, chair in candidates(node):
            new_cost = cost + extra
            if new_cost >= best_cost:
                continue
            steps += 1
            if steps > budget:
                return True
            seat[node] = chair
            used.add(chair)
            stop = extend(k + 1, new_cost)
            del seat[node]
            used.discard(chair)
            if stop:
                return True
        return False

    extend(0, 0.0)
    return best_seat


def closest_seating(
    pattern: nx.Graph,
    reference: dict[int, int],
    chip: Chip,
    upcoming: list[tuple[int, int, float]] = (),
) -> dict[int, int]:
    """The cheapest seating found (see module docstring). The pattern must be one
    that fit.fits() accepted, so a seating is known to exist."""
    seat = search_seating(pattern, chip, reference, upcoming, CLOSEST_BUDGET) or search_seating(pattern, chip)
    if seat is None:
        raise ValueError("pattern has no seating; check fits() first")
    return seat
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_seating.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/seating.py period_router/tests/test_seating.py
git commit -m "period router: budgeted seating search with cross-period look-ahead"
```

---

### Task 6: Does a stretch fit, and how far does the period go?

Instant rejections first: too many partners for any chair, an odd loop on a chip that has only even loops, or a loop shorter than the chip's shortest. These also make good explanations in the writeup. Then the seating search. The period's end is found by **binary search**, which is valid because fitting is monotone (if gates 10–30 fit, gates 10–29 do too). `lo` only ever moves to an end that really fits, so the answer is always a stretch that fits, even if the search budget runs out somewhere.

**Files:**
- Create: `period_router/fit.py`
- Test: `period_router/tests/test_fit.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_fit.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_fit.py -v`
Expected: FAIL with `No module named 'period_router.fit'`

- [ ] **Step 3: Implement**

`period_router/fit.py`:

```python
"""Does a stretch of the program fit on the chip with zero SWAPs?

"Fits" means: there is a seating where every interacting pair sits on
neighbouring physical qubits. In graph terms, the interaction graph is
subgraph-monomorphic to the chip. Monomorphic, not isomorphic: two qubits that
never interact are allowed to sit next to each other.
"""

import networkx as nx

from .chip import Chip
from .program import interaction_graph
from .seating import search_seating


def cannot_fit(pattern: nx.Graph, chip: Chip) -> bool:
    """Instant proofs that a pattern cannot fit. False means 'not ruled out'."""
    if pattern.number_of_edges() == 0:
        return False
    if pattern.number_of_nodes() > chip.graph.number_of_nodes():
        return True
    if max(degree for _, degree in pattern.degree()) > chip.max_degree:
        return True  # someone has more partners than any chair has neighbours
    if chip.bipartite and not nx.is_bipartite(pattern):
        return True  # an odd loop cannot be drawn on a chip with only even loops
    if nx.girth(pattern) < chip.girth:
        return True  # a loop shorter than the chip's shortest loop
    return False


def fits(pattern: nx.Graph, chip: Chip) -> bool:
    """True only if a seating was actually found. If the search budget runs out,
    the answer is False: the period just ends earlier, which is always safe."""
    if pattern.number_of_edges() == 0:
        return True
    if cannot_fit(pattern, chip):
        return False
    return search_seating(pattern, chip) is not None


def longest_fitting_end(program: list[tuple], start: int, chip: Chip) -> int:
    """Largest end such that program[start:end] fits. Always at least start + 1.

    Binary search works because fitting is monotone: if a stretch fits, every
    shorter stretch from the same start fits too. It only ever moves `lo` onto
    an end that fits(), so the answer is always a stretch that really fits.
    """
    lo, hi = start + 1, len(program)  # program[start:lo] is a single op, which always fits
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(interaction_graph(program[start:mid]), chip):
            lo = mid
        else:
            hi = mid - 1
    return lo
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_fit.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/fit.py period_router/tests/test_fit.py
git commit -m "period router: fit checks and binary search for the period end"
```

---

### Task 7: Reshuffle (token swapping)

Phase 1 is greedy: repeatedly make the SWAP that most reduces the target-holders' total distance to their targets. Phase 2 runs only when phase 1 is stuck. It fills the leaves of a spanning tree one at a time and locks them, which always finishes. The fuzz test runs 800 random reshuffles over four chips and checks that every target is reached using only real wires.

**Files:**
- Create: `period_router/token_swap.py`
- Test: `period_router/tests/test_token_swap.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_token_swap.py`:

```python
import random

from starter_kit import build_hardware_graph

from period_router.chip import make_chip
from period_router.layout import Layout
from period_router.random_programs import unseen_chips
from period_router.token_swap import token_swap


def test_worked_example(chip):
    layout = Layout({0: 5, 1: 6, 2: 7, 3: 11})
    swaps = token_swap(layout, {0: 7, 2: 6, 3: 11}, chip)
    assert swaps == [(5, 6), (6, 7)]
    assert layout.where == {0: 7, 1: 5, 2: 6, 3: 11}


def test_nothing_to_do(chip):
    layout = Layout({0: 5, 1: 6})
    assert token_swap(layout, {0: 5}, chip) == []


def test_random_reshuffles_always_arrive_using_real_wires():
    chips = {"challenge": build_hardware_graph(), **unseen_chips()}
    rng = random.Random(0)
    for graph in chips.values():
        chip = make_chip(graph)
        chairs = list(graph.nodes)
        for _ in range(200):
            count = rng.randint(1, len(chairs))
            qubits = rng.sample(range(40), count)
            layout = Layout(dict(zip(qubits, rng.sample(chairs, count))))
            movers = rng.sample(qubits, rng.randint(1, count))
            target = dict(zip(movers, rng.sample(chairs, len(movers))))
            swaps = token_swap(layout, target, chip)
            assert all(graph.has_edge(p, q) for p, q in swaps)
            assert all(layout.where[q] == chair for q, chair in target.items())
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_token_swap.py -v`
Expected: FAIL with `No module named 'period_router.token_swap'`

- [ ] **Step 3: Implement**

`period_router/token_swap.py`:

```python
"""Reshuffle: neighbour SWAPs that move some qubits onto target chairs.

Qubits without a target, and empty chairs, can end up anywhere.

Phase 1 is greedy: keep making the SWAP that brings target-holders closest to
their targets in total, while some SWAP still helps. It is short but can get
stuck, e.g. when a qubit must pass through one that is already in place.

Phase 2 always finishes: fill the leaves of a spanning tree one at a time, then
lock them. It is longer, so it only runs when phase 1 gets stuck. Measured: the
reshuffles the period router actually keeps are almost all small (0-3 SWAPs),
so a cleverer phase 2 would barely change the score.
"""

import networkx as nx

from .chip import Chip
from .layout import Layout


def token_swap(layout: Layout, target: dict[int, int], chip: Chip) -> list[tuple[int, int]]:
    """Move every logical qubit in `target` onto its chair. Mutates `layout`
    and returns the SWAPs made, in order."""
    swaps: list[tuple[int, int]] = []
    dist = chip.dist

    def swap(p: int, q: int) -> None:
        if layout.occupant.get(p) is None and layout.occupant.get(q) is None:
            return  # swapping two empty chairs changes nothing
        layout.swap(p, q)
        swaps.append((p, q))

    def gain(p: int, q: int) -> int:
        a, b = layout.occupant.get(p), layout.occupant.get(q)
        total = 0
        if a in target:
            total += dist[p][target[a]] - dist[q][target[a]]
        if b in target:
            total += dist[q][target[b]] - dist[p][target[b]]
        return total

    # Phase 1: greedy
    while True:
        best_edge, best_gain = None, 0
        for p, q in chip.graph.edges:
            g = gain(p, q)
            if g > best_gain:
                best_edge, best_gain = (p, q), g
        if best_edge is None:
            break
        swap(*best_edge)

    if all(layout.where[logical] == chair for logical, chair in target.items()):
        return swaps

    # Phase 2: spanning-tree leaf filling
    remaining = nx.Graph(nx.bfs_tree(chip.graph, chip.center))
    wanted_at = {chair: logical for logical, chair in target.items()}
    while remaining.number_of_nodes():
        leaf = min(v for v in remaining if remaining.degree(v) <= 1)
        wanted = wanted_at.get(leaf)
        if wanted is not None:
            source = layout.where[wanted]
        else:  # the leaf needs anything that has no target: an empty chair or a free qubit
            free = [v for v in remaining if layout.occupant.get(v) not in target]
            lengths = nx.single_source_shortest_path_length(remaining, leaf)
            source = min(free, key=lambda v: (lengths[v], v))
        path = nx.shortest_path(remaining, source, leaf)
        for p, q in zip(path, path[1:]):
            swap(p, q)
        remaining.remove_node(leaf)
    return swaps
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_token_swap.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/token_swap.py period_router/tests/test_token_swap.py
git commit -m "period router: token swapping with an always-finishing fallback"
```

---

### Task 8: Gate-by-gate lookahead router

Used two ways: to route a period without reshuffling, and as a standalone router in the portfolio. Every SWAP it makes moves one of the gate's two qubits one hop closer to the other, so a gate at distance *d* takes exactly *d*−1 SWAPs and the loop always ends. The lookahead only chooses *which* of those SWAPs to make. There is no "front layer" as in SABRE, because rule 1 fixes the gate order.

**Files:**
- Create: `period_router/lookahead.py`
- Test: `period_router/tests/test_lookahead.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_lookahead.py`:

```python
from starter_kit import BENCHMARKS, score_summary

from period_router.layout import Layout
from period_router.lookahead import route_gate, route_lookahead


def test_route_gate_needs_exactly_distance_minus_one_swaps(chip):
    layout = Layout({0: 0, 1: 19})  # 7 hops apart
    swaps = route_gate(layout, 0, 1, [], chip)
    assert len(swaps) == 6
    assert all(q in chip.neighbors[p] for p, q in swaps)
    assert chip.dist[layout.where[0]][layout.where[1]] == 1


def test_route_gate_does_nothing_for_neighbours(chip):
    layout = Layout({0: 5, 1: 6})
    assert route_gate(layout, 0, 1, [], chip) == []


def test_route_lookahead_is_valid_on_every_benchmark(hw, chip):
    for name, program in BENCHMARKS.items():
        placement = {q: q for q in {q for op in program for q in op[1:]}}
        routed = route_lookahead(program, placement, chip)
        result = score_summary(program, hw, placement, routed)
        assert result["valid"], (name, result["message"])
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_lookahead.py -v`
Expected: FAIL with `No module named 'period_router.lookahead'`

- [ ] **Step 3: Implement**

`period_router/lookahead.py`:

```python
"""Gate-by-gate router with lookahead: the fallback for stretches that don't fit.

For a gate whose qubits are apart, every SWAP it makes moves one of them one
hop closer to the other, so routing a gate always ends. When several SWAPs do
that, it picks the one that leaves the next few gates closest together.
"""

from .chip import Chip
from .layout import Layout
from .program import upcoming_pairs

LOOKAHEAD = 20  # how many upcoming gates to consider
DECAY = 0.8  # each later gate counts this much less than the one before


def upcoming_cost(layout: Layout, upcoming: list[tuple[int, int, float]], chip: Chip) -> float:
    return sum(weight * chip.dist[layout.where[a]][layout.where[b]] for a, b, weight in upcoming)


def route_gate(layout: Layout, a: int, b: int, upcoming: list[tuple[int, int, float]], chip: Chip) -> list[tuple[int, int]]:
    """SWAPs that make logical a and b neighbours. Mutates `layout`."""
    swaps = []
    dist = chip.dist
    while dist[layout.where[a]][layout.where[b]] > 1:
        pa, pb = layout.where[a], layout.where[b]
        gap = dist[pa][pb]
        options = [
            (mover, step)
            for mover, other in ((pa, pb), (pb, pa))
            for step in sorted(chip.graph.neighbors(mover))
            if dist[step][other] == gap - 1
        ]

        def score(option: tuple[int, int]) -> tuple[float, tuple[int, int]]:
            layout.swap(*option)
            value = upcoming_cost(layout, upcoming, chip)
            layout.swap(*option)  # undo
            return value, option

        best = min(options, key=score)
        layout.swap(*best)
        swaps.append(best)
    return swaps


def route_span(program: list[tuple], start: int, end: int, layout: Layout, chip: Chip) -> list[tuple]:
    """Route program[start:end] gate by gate. Looks past `end` for lookahead. Mutates `layout`."""
    routed = []
    for index in range(start, end):
        op = program[index]
        if op[0] == "2Q":
            upcoming = upcoming_pairs(program, index + 1, LOOKAHEAD, DECAY)
            for p, q in route_gate(layout, op[1], op[2], upcoming, chip):
                routed.append(("SWAP", p, q))
        routed.append(layout.emit(op))
    return routed


def route_lookahead(program: list[tuple], placement: dict[int, int], chip: Chip) -> list[tuple]:
    """The whole program, gate by gate."""
    return route_span(program, 0, len(program), Layout(placement), chip)
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_lookahead.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/lookahead.py period_router/tests/test_lookahead.py
git commit -m "period router: gate-by-gate lookahead router"
```

---

### Task 9: Starting seatings

Two seeds. `initial_placement` seats the first period around the chip's centre, with look-ahead, then puts every other qubit on the free chair nearest the partners it talks to most. `label_order_placement` puts qubit *k* on chair *k*. It sounds naive, but measured: without it, "local" programs on a ring chip scored up to **1.53×** the baseline, because the baseline uses exactly this seating and it happens to suit programs whose labels follow the wiring.

**Files:**
- Create: `period_router/initial.py`
- Test: `period_router/tests/test_initial.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_initial.py`:

```python
import pytest

from starter_kit import BENCHMARKS

from period_router.initial import initial_placement, label_order_placement
from period_router.program import logical_qubits


def test_every_used_qubit_gets_its_own_chair(chip):
    for name, program in BENCHMARKS.items():
        placement = initial_placement(program, chip)
        assert set(placement) == set(logical_qubits(program)), name
        assert len(set(placement.values())) == len(placement), name
        assert set(placement.values()) <= set(chip.neighbors), name


def test_missing_labels_are_not_placed(chip):
    # qaoa_random never uses qubits 4 and 7; placing them makes the scorer reject the answer
    placement = initial_placement(BENCHMARKS["qaoa_random"], chip)
    assert 4 not in placement and 7 not in placement


def test_first_period_is_seated_with_every_pair_adjacent(chip):
    placement = initial_placement(BENCHMARKS["chain_trotter"], chip)
    for _, a, b in BENCHMARKS["chain_trotter"]:
        assert placement[b] in chip.neighbors[placement[a]]


def test_too_many_qubits_is_an_error(chip):
    program = [("2Q", q, q + 1) for q in range(21)]  # 22 qubits on a 20-qubit chip
    with pytest.raises(ValueError):
        initial_placement(program, chip)


def test_label_order_placement(chip):
    assert label_order_placement([("2Q", 3, 7), ("1Q", 9)], chip) == {3: 0, 7: 1, 9: 2}
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_initial.py -v`
Expected: FAIL with `No module named 'period_router.initial'`

- [ ] **Step 3: Implement**

`period_router/initial.py`:

```python
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
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_initial.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/initial.py period_router/tests/test_initial.py
git commit -m "period router: centre-out and label-order starting seatings"
```

---

### Task 10: The period router

For each period, both options are built on copies of the layout and compared by the **real score of everything emitted so far** (`solver.cost.cost`, which includes depth), so the choice is never worse locally. The worked example in the test is the one from the design discussion: a line 0–1–2–3, then 3–0 and 0–2. It costs 2 SWAPs, depth 6, score 5.0. The baseline scores 8.5 on it.

**Files:**
- Create: `period_router/periods.py`
- Test: `period_router/tests/test_periods.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_periods.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_periods.py -v`
Expected: FAIL with `No module named 'period_router.periods'`

- [ ] **Step 3: Implement**

`period_router/periods.py`:

```python
"""The period router.

Cut the program into periods: the longest stretches that fit with zero SWAPs.
For each period, compare two ways of handling it and keep the cheaper:
  reseat - reshuffle to the closest seating where the whole period fits, or
  route  - leave the seating alone and route the period gate by gate.
"Cheaper" is the real score (SWAPs + 0.5 x depth) of everything emitted so far.
"""

from solver.cost import cost

from .chip import Chip
from .fit import longest_fitting_end
from .layout import Layout
from .lookahead import DECAY, LOOKAHEAD, route_span
from .program import interaction_graph, upcoming_pairs
from .seating import closest_seating
from .token_swap import token_swap


def reseat_span(program: list[tuple], start: int, end: int, layout: Layout, chip: Chip) -> list[tuple]:
    """Reshuffle to the closest seating that fits program[start:end], then emit it. Mutates `layout`."""
    pattern = interaction_graph(program[start:end])
    routed = []
    if pattern.number_of_edges():
        upcoming = upcoming_pairs(program, end, LOOKAHEAD, DECAY)
        target = closest_seating(pattern, layout.where, chip, upcoming)
        routed = [("SWAP", p, q) for p, q in token_swap(layout, target, chip)]
    routed.extend(layout.emit(op) for op in program[start:end])
    return routed


def route_periods(program: list[tuple], placement: dict[int, int], chip: Chip, stats: dict | None = None) -> list[tuple]:
    layout = Layout(placement)
    routed: list[tuple] = []
    start = 0
    while start < len(program):
        end = longest_fitting_end(program, start, chip)

        reseat_layout = layout.copy()
        reseated = reseat_span(program, start, end, reseat_layout, chip)
        route_layout = layout.copy()
        gate_by_gate = route_span(program, start, end, route_layout, chip)

        if cost(routed + reseated) <= cost(routed + gate_by_gate):
            routed += reseated
            layout = reseat_layout
            choice = "reseat"
        else:
            routed += gate_by_gate
            layout = route_layout
            choice = "route"

        if stats is not None:
            stats.setdefault("periods", []).append((end - start, choice))
        start = end
    return routed
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_periods.py -v`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add period_router/periods.py period_router/tests/test_periods.py
git commit -m "period router: reseat-or-route per period, chosen by real score"
```

---

### Task 11: `solve()`, the portfolio

Every router, from every seed, refined with the forward-backward trick while it lowers the score (at most 5 rounds). Every candidate is validated before it can win; an invalid one raises, because it means a bug. The random-program test asserts validity *and* a score at or below the baseline's on four chips, three styles and three seeds each.

**Files:**
- Create: `period_router/solve.py`
- Modify: `period_router/__init__.py` (whole file)
- Test: `period_router/tests/test_solve.py`

- [ ] **Step 1: Write the failing test**

`period_router/tests/test_solve.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m uv run --with pytest pytest period_router/tests/test_solve.py -v`
Expected: FAIL with `cannot import name 'solve' from 'period_router'`

- [ ] **Step 3: Implement**

`period_router/solve.py`:

```python
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
```

Replace all of `period_router/__init__.py` with:

```python
"""Period router for the QSITE 2026 Computational Track.

    from period_router import solve
    placement, routed_program = solve(program, hardware_graph)
"""

from .solve import solve

__all__ = ["solve"]
```

- [ ] **Step 4: Run it to see it pass**

Run: `python -m uv run --with pytest pytest period_router/tests/test_solve.py -v`
Expected: `4 passed`

- [ ] **Step 5: Run the whole suite**

Run: `python -m uv run --with pytest pytest period_router/tests -q`
Expected: `48 passed`

- [ ] **Step 6: Commit**

```bash
git add period_router/solve.py period_router/__init__.py period_router/tests/test_solve.py
git commit -m "period router: solve() portfolio with forward-backward refinement"
```

---

### Task 12: Results tables for the writeup

No test. This is a report, and every number in it is checked for validity as it is produced.

**Files:**
- Create: `period_router/compare.py`

- [ ] **Step 1: Implement**

`period_router/compare.py`:

```python
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
```

- [ ] **Step 2: Run it**

Run: `python -m uv run python -m period_router.compare`
Expected (takes about 45 seconds):

```
benchmark         baseline    ours  swaps  depth
ghz_star              14.0     8.0      3     10
chain_trotter         15.0     4.5      0      9
ladder_trotter        35.5     9.0      5      8
qaoa_random           39.0    15.0      9     12
dense_random         122.0    49.5     34     31
vqe_layers            58.0     3.0      0      6
TOTAL                283.5    89.0

chip           programs  mean ours/baseline  worst
challenge            30               0.446  0.622
grid_4x5             30               0.321  0.526
ring_20              30               0.462  0.656
regular3_20          30               0.293  0.532
```

followed by the elapsed time.

- [ ] **Step 3: Commit**

```bash
git add period_router/compare.py
git commit -m "period router: comparison tables for the writeup"
```

---

### Task 13: Use it in the notebook

Cell 20 of `starter.ipynb` (id `f8cdd038`) is the organisers' `def solve(...): raise NotImplementedError` stub. It becomes an import plus a loop over all six benchmarks. The script below edits only that cell. It writes JSON exactly the way Jupyter does, so every other cell stays byte-identical, and then runs cells 1 and 20 to prove the edit works.

**Files:**
- Modify: `starter.ipynb` (cell 20 only)
- Temporary: `wire_notebook.py` (deleted in Step 3)

- [ ] **Step 1: Create the one-off script**

`wire_notebook.py` (in the track folder):

```python
"""One-off: point starter.ipynb's submission cell at period_router, then run it.

Run from the Computational Track folder, then delete this file.
"""

import json
from pathlib import Path

SUBMISSION_CELL_ID = "f8cdd038"  # cell 20, "def solve(...): raise NotImplementedError"
SETUP_CELL_INDEX = 1  # imports, sys.path, GRAPH

NEW_SOURCE = '''from period_router import solve

# Score every benchmark with the period router, next to the baseline.
baseline_total = our_total = 0.0
for name, program in BENCHMARKS.items():
    bl_pl, bl_rt = baseline_solve(program, GRAPH)
    baseline = score_summary(program, GRAPH, bl_pl, bl_rt)
    placement, routed = solve(program, GRAPH)
    result = score_summary(program, GRAPH, placement, routed)
    if not result["valid"]:
        raise AssertionError(f"{name}: {result['message']}")
    baseline_total += baseline["score"]
    our_total += result["score"]
    print(f"{name:16s} baseline {baseline['score']:6.1f}   ours {result['score']:6.1f}"
          f"   ({result['swap_count']} swaps, depth {result['depth']})")
print(f"{'TOTAL':16s} baseline {baseline_total:6.1f}   ours {our_total:6.1f}")
'''

path = Path("starter.ipynb")
notebook = json.loads(path.read_text(encoding="utf-8"))
cell = next(c for c in notebook["cells"] if c.get("id") == SUBMISSION_CELL_ID)
cell["source"] = NEW_SOURCE.splitlines(keepends=True)
cell["outputs"] = []
cell["execution_count"] = None
# indent=1 + ensure_ascii=False is how Jupyter saves, so every other cell stays byte-identical
path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("submission cell replaced; running cells 1 and 20 as Jupyter would:\n")

setup = "".join(notebook["cells"][SETUP_CELL_INDEX]["source"])
exec(compile(setup + "\n" + NEW_SOURCE, "starter.ipynb", "exec"), {})
```

- [ ] **Step 2: Run it**

Run: `python -m uv run python wire_notebook.py`
Expected (the harmless `autograd` warning may appear just before `Hardware graph`):

```
submission cell replaced; running cells 1 and 20 as Jupyter would:

Hardware graph: 20 qubits, 23 connections
PennyLane version: 0.45.1
ghz_star         baseline   14.0   ours    8.0   (3 swaps, depth 10)
chain_trotter    baseline   15.0   ours    4.5   (0 swaps, depth 9)
ladder_trotter   baseline   35.5   ours    9.0   (5 swaps, depth 8)
qaoa_random      baseline   39.0   ours   15.0   (9 swaps, depth 12)
dense_random     baseline  122.0   ours   49.5   (34 swaps, depth 31)
vqe_layers       baseline   58.0   ours    3.0   (0 swaps, depth 6)
TOTAL            baseline  283.5   ours   89.0
```

- [ ] **Step 3: Delete the script and check the diff is only cell 20**

Delete `wire_notebook.py`, then run: `git diff --stat starter.ipynb`
Expected: `1 file changed, 16 insertions(+), 58 deletions(-)`

- [ ] **Step 4: Commit**

```bash
git add starter.ipynb
git commit -m "notebook: submission cell uses period_router.solve on all benchmarks"
```

---

### Task 14: Final verification and hand-off

- [ ] **Step 1: Whole suite**

Run: `python -m uv run --with pytest pytest period_router/tests -q`
Expected: `48 passed`

- [ ] **Step 2: Tree is clean apart from the lock file**

Run: `git status -sb`
Expected: `## period-router` and, at most, `?? uv.lock`.

- [ ] **Step 3: Decide about `uv.lock` with the team**

`uv.lock` pins the exact package versions this plan was measured with (networkx 3.7 etc.). The Scientific Track already commits its own. If the team agrees:

```bash
git add uv.lock
git commit -m "computational track: lock package versions"
```

If not, leave it untracked.

- [ ] **Step 4: Hand-off**

Do not push or open a PR without the team. Share: the Task 12 tables, and Appendix A (which is the writeup's "method and evidence" section almost as-is).

---

## Appendix A: Every design choice, and the measurement behind it

"Random" below means the 120-program suite in `compare.py`: 3 styles × 10 seeds × 4 chips, scored as our score ÷ baseline score, lower is better.

| Choice | Measured | Decision |
|---|---|---|
| Fit check engine | networkx VF2 took 48 s on one 16-qubit program on the 4×5 grid and can't be interrupted. The budgeted search: slowest program 0.2 s, and it only says "fits" when it found a seating | Own budgeted search (Task 5) |
| Label-order second seed | Without it: worst random case 1.53× baseline (ring chip, "local" programs). With it: 0.97× | Keep (Task 9) |
| Tie-break between equally close chairs | (hops, −degree, chair): 0.4468; (hops, chair): 0.4444. No real difference | Simpler one |
| `UPCOMING_WEIGHT` (look-ahead across periods) | 0 → bench 109.0, random 0.444 · 0.25 → 104.0, 0.420 · 0.5 → 104.0, 0.419 · **1.0 → 104.0, 0.417** · 2.0 → 110.5, 0.417 | 1.0: one hop later counts the same as one hop now; not tuned to benchmarks |
| Token-swap phase 2 | Spanning-tree fill: 2.26× the ⌈D/2⌉ lower bound. Articulation-point locking: 2.79×. Identical scores; tree 5× faster. Of reshuffles that are *kept*, 355 of 450 need 0–3 SWAPs; of reshuffles needing 10+, 503 of 517 lose to gate-by-gate routing anyway | Tree fill; a better phase 2 would barely move the score |
| Forward-backward refinement | off → 104.0, 0.4169 · 1 round → 103.5, 0.3857 · 2 → 95.5, 0.3810 · 3 → 89.0, 0.3800 · **adaptive (≤5, stop when no gain) → 89.0, 0.3782** | Adaptive: no round count to tune |
| `CLOSEST_BUDGET` | 20000 → 0.3782, 30.1 s on a 20-qubit × 150-gate ring program · 5000 → 0.3792, 13.0 s · **2000 → 0.3804, 7.7 s** · 500 → 0.3840 (bench 92.0), 4.6 s | 2000: 0.6% worse, 4× faster |

## Appendix B: Known limits (not part of this plan)

- `ghz_star` scores 8.0; `plan.md` cites 6.5 (2 SWAPs) as its proven optimum.
- Reshuffles use about 2.3× the SWAP lower bound on large reshuffles. A proper approximate token swapper (Miltzow et al., 2016, the basis of Qiskit's `ApproximateTokenSwapper`) could let reseating win more often.
- Periods must fit *perfectly*. Allowing a few routed pairs inside a period ("relaxed periods", chosen by dynamic programming over cut points) is the natural next step for dense programs.
- Runtime: under 0.25 s per benchmark; up to about 8 s for a 20-qubit, 150-gate program on a ring chip.
- Assumes a connected hardware graph and no more logical qubits than physical ones (`make_chip` and `initial_placement` raise `ValueError` otherwise).
- `solve()` is a portfolio. Routers from `plan.md` (A*, windowed exact routing) can join it by being added to `ROUTERS`, and placement methods by being added to the seed list.
