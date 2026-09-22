# Plan: author `Computational Track/plan.md` — a self-implementation guide for the solver

## Context

You are building a submission for the QSITE 2026 Quantum Coalition Computational Track: a
`solve(program, hardware_graph)` that places logical qubits on a 20-node chip and inserts SWAPs so
every 2Q gate lands on a hardware edge. Score is `Σ (swaps + 0.5 × depth)` over six benchmarks,
lower is better. The baseline scores **283.5**; a zero-swap solution would score **24.0**; good
independent attempts land around **75–85**.

You want to write the solver yourself. So the deliverable of *this* task is not code — it is a
single document, `Computational Track/plan.md`, that walks you stage by stage: what each module is
for, what its interface should be, the algorithm in words, the hints that save hours, the traps that
cause silent disqualification, and a measurable checkpoint after every stage so you always know
whether you are winning.

**The only file created by this plan is `Computational Track/plan.md`.** No solver code is written.
The content below is the document to be written.

---

## Document to write: `Computational Track/plan.md`

### Front matter

- State the goal, the scoring formula, and the three numbers that matter: baseline **283.5**,
  all-zero-swap floor **24.0**, target **< 90**.
- State the ground rule for the reader: every stage ends with a runnable checkpoint and an expected
  number. If the number is wrong, do not advance.
- Note that the grader is `starter_kit/scorer.py` and it is only ever run to *report*, never inside
  the search loop — this is what keeps the algorithm general rather than benchmark-fitted.

### Section 0 — Scaffolding

Target layout to state up front:

```
Computational Track/
├─ starter.ipynb          # cell 20 becomes a 3-line wrapper
├─ solver/
│  ├─ __init__.py         # exports solve()
│  ├─ cost.py             # own score + validator
│  ├─ interaction.py      # program → interaction graph
│  ├─ embed.py            # VF2 zero-SWAP solver
│  ├─ placement.py        # spectral seed + local search
│  ├─ router.py           # lookahead heuristic router
│  ├─ exact.py            # A* with admissible bounds
│  ├─ windows.py          # windowed exact routing
│  └─ driver.py           # portfolio + time budget
├─ tests/
└─ writeup.md
```

Hints to include:
- Notebook cell 1 already does `NOTEBOOK_DIR = Path.cwd()` plus a `sys.path` insert, so a sibling
  `solver/` package imports with no extra work.
- Keep modules free of imports outside `solver/` (other than `networkx`) so the whole package can be
  flattened into one notebook cell if judges run only the `.ipynb`.
- The interpreter is `/opt/miniconda3/envs/q-site-hacks-2026/bin/python`. Plain `python3` has no
  `networkx`.

### Section 1 — `cost.py` (build first, everything depends on it)

Purpose: score and validate without touching the grader.

Guide the reader to write four functions, described by interface and behaviour only:

1. **`depth(routed_program)`** — ASAP layering. Walk ops in order; skip `1Q` entirely (the grader
   does — `scorer.py:78-79`); for each `2Q`/`SWAP`, its layer is `1 + max(last layer of either
   wire)`; record the new layer for both wires. Return the count of layers. Hint: a single dict from
   physical qubit → last layer index is the whole algorithm; you never need to materialise the
   layers to get the number.
2. **`score(routed_program)`** — `swaps + 0.5 * depth`. Three lines.
3. **`validate(program, graph, placement, routed_program)`** — reimplement the three rules:
   every `2Q`/`SWAP` on a real edge; replay SWAPs through a physical→logical map and confirm the
   translated op list equals `program` **exactly, in order**; placement covers exactly the used
   logical qubits and is injective.
4. **`used_logical_qubits(program)`** — the set of qubits actually appearing, sorted.

Traps to spell out in the document, each with the reason:
- **`qaoa_random` is missing logical labels 4 and 7.** `_random_pairs(12, 18, seed=7)` samples from
  `range(12)` but only 10 labels survive. If placement is built from `range(n_qubits)` instead of
  from the labels that appear, `scorer.py:15` rejects it and the benchmark scores ∞.
- **Tuple orientation is load-bearing.** `scorer.py:50` appends `("2Q", left, right)` in the order
  the physical qubits appear. Sorting the pair, or emitting `(j, i)` where the program said
  `(i, j)`, fails the equality check on line 68. This invalidates 5 of the 6 benchmarks.
- **Never mutate the caller's `program` list**, and never return a placement dict you mutated during
  routing. `baseline_routing.py:45` dodges this by recomputing `identity_placement` fresh — a
  mutated placement scores fine on a warm run and ∞ on a cold one.
- **A SWAP onto an empty physical qubit writes `None` into the map** (`scorer.py:41-43`). The
  `None` key then passes the membership guard on line 48 and produces a misleading error much later.
  Your own validator should reject this early with a clear message.

Checkpoint: run your `score` against `starter_kit.scorer.core_score` on the baseline's output for all
six benchmarks. **They must agree to the digit, and the total must be 283.5.** If they disagree,
stop — every later measurement is meaningless.

### Section 2 — `interaction.py`

Purpose: turn a program into the structure placement reasons about.

- **`interaction_graph(program)`** — nodes are used logical qubits, an edge for every distinct pair
  appearing in a `2Q`. Hint: also store a **weight** = how many times that pair occurs. Weight is
  what makes placement care about hot pairs; `vqe_layers` repeats its layer 3×, so weights are 3.
- **`gate_list(program)`** — just the `2Q` ops, in order. This is what the router consumes; 1Q ops
  are passed through and never affect score.
- A small `stats()` for the writeup: qubit count, edge count, max degree, component count.

Checkpoint: print the table for all six. It should read:

```
ghz_star        lq= 8  iedges= 7   maxdeg=7   comps=1
chain_trotter   lq=10  iedges= 9   maxdeg=2   comps=1
ladder_trotter  lq=12  iedges=16   maxdeg=3   comps=1
qaoa_random     lq=10  iedges=15              comps=1
dense_random    lq=14  iedges=37              comps=1
vqe_layers      lq=16  iedges=15   maxdeg=2   comps=1
```

Point out what this table already tells you: `ghz_star` has a degree-7 node and the chip's maximum
degree is 3, so it *cannot* be swap-free — that is a proof, not a guess. And every benchmark is a
single component, so component-splitting is free generality that buys nothing here; include it,
don't advertise it.

### Section 3 — `embed.py` — the zero-SWAP solver

Purpose: when the interaction graph fits inside the chip, find the placement that needs no SWAPs at
all, and win that benchmark outright.

- The right notion is **subgraph monomorphism**, not isomorphism: you need every interaction edge to
  land on a hardware edge, but you do not care that the chip has extra edges between your qubits.
- Hint on the API: `networkx.algorithms.isomorphism.GraphMatcher(hardware, interaction)` exposes
  `subgraph_monomorphisms_iter()`. Note the argument order and note that the yielded mapping is
  **physical → logical**, so it must be inverted before returning.
- Cheap necessary conditions to test *before* calling VF2, because they are instant and they explain
  the failures in the writeup: interaction max degree ≤ hardware max degree; if the hardware is
  bipartite (this one is), an odd cycle in the interaction graph is fatal; interaction girth ≥
  hardware girth (this chip's girth is 6).
- Take the **first** mapping and stop. Any monomorphism gives zero swaps; they are all equally
  optimal on swaps, and depth is already fixed by the program order.

Checkpoint: exactly two benchmarks should succeed — `chain_trotter` → **4.5** and `vqe_layers` →
**3.0**, both zero swaps. Those two are then finished forever; no later stage can improve them.
Score after this stage, with the baseline still handling the other four: **~244**.

Also record *why* the other four fail, because these lines go straight into the presentation:
`ghz_star` degree 7 > 3; `ladder_trotter` girth 4 < 6; `qaoa_random` and `dense_random` contain
triangles and the chip is bipartite.

### Section 4 — `router.py` — the heuristic router

Purpose: given a placement, emit a valid routed program. This is the workhorse; placement search
will call it thousands of times, so it must be fast and it must never produce an invalid output.

Structure to guide toward:
- Precompute the **all-pairs shortest-path distance matrix** of the hardware once and pass it in.
  Hint: `dict(nx.all_pairs_shortest_path_length(G))` is fine at n=20; do it once, never per call.
- Maintain `pos` (logical → physical) and its inverse as you go.
- Walk the gate list **strictly in order** — this is the constraint that rules out SABRE's front
  layer. For gate `(a, b)`: if `pos[a]` and `pos[b]` are adjacent, emit the 2Q and move on.
  Otherwise insert SWAPs until they are adjacent, then emit the 2Q.
- For choosing SWAPs, the key idea: consider only swaps on hardware edges that **touch a physical
  qubit currently holding a logical qubit that appears in some remaining gate**. A swap touching
  nothing live cannot change any remaining distance and can never help. This prune is both a
  correctness-preserving speedup and a good slide.
- Score each candidate swap by how much it reduces `distance(pos[a], pos[b])` for the current gate,
  **plus a decayed lookahead term** over the next *k* gates (k ≈ 20). Hint on weighting: the current
  gate must dominate, or the router stalls; something like `current + 0.5 × (mean over lookahead)`
  is the standard shape. Tell the reader to derive the weight from the argument "the current gate is
  the only one that is blocking", not by sweeping values against the scorer.
- Two things worth trying and *measuring*, not assuming: (a) moving both endpoints toward each other
  versus moving only one; (b) when the swap count for a gate would be odd, the parity trick that
  occasionally saves 0.5 of depth. Flag explicitly that bidirectional routing is **not** uniformly
  better — it improves the total but makes `ghz_star` worse (7 swaps → 10). It belongs in the
  portfolio as a scored candidate, not as the default.

Checkpoint: run with the identity placement and confirm it beats the baseline's 283.5 comfortably,
and that `validate` passes on all six. Expect roughly **150–200** at this stage — placement is still
naive, so most of the win is still ahead.

### Section 5 — `placement.py` — the bigger lever

Purpose: choose the starting map. Measured ablation says placement is worth roughly **1.7×** what
routing is worth, so this is where the remaining score lives.

Guide toward a **seed + improve** structure:

**Seeds** (generate several, they are cheap):
1. **Spectral.** Compute the Fiedler vector of the weighted interaction graph and sort logical
   qubits by their coordinate; compute a linear order of the hardware the same way (or use the
   chip's Hamiltonian path, below); zip them together. This is Hall's 1970 quadratic placement and
   it is the one that "makes physical sense" — it minimises squared edge length. Hint:
   `nx.fiedler_vector` and `nx.spectral_ordering` exist.
2. **Hamiltonian path.** This chip has one:
   `[3,2,1,0,4,5,6,7,11,10,9,8,12,13,14,15,19,18,17,16]`. Laying the interaction graph's BFS or
   Cuthill–McKee order along it is a strong, very cheap seed. State clearly that the path is used as
   a *placement line*, and that reducing routing to sorting along it is a dead end — a covering
   sweep is only optimal if gates may be consumed in any order, and `scorer.py:68` forbids that. It
   was tried and scored 4040 versus 84.
3. **Degree-greedy.** Put the highest-degree logical qubit on a highest-degree, most-central
   physical qubit, then place its neighbours outward by BFS. This is the seed that does well on
   star-shaped programs.
4. **Random restarts** to fill out the portfolio.

**Improve**: local search over the seed. Neighbourhood = swap two logical qubits' physical
assignments, or move one logical qubit to an unused physical qubit. Accept if the *cost function
improves*. Two choices of cost, and the document should recommend trying both:
- **Cheap proxy**: `Σ over interaction edges of weight × distance(pos[u], pos[v])`. Evaluates in
  microseconds, so you can do tens of thousands of steps.
- **True cost**: actually run the router and take its score. Far more accurate, far slower. Use it
  only for a final polish pass on the best few candidates.

Hint on the standard trick: optimise on the proxy for the bulk of the budget, then re-rank the top
handful under the true router cost. State the guarantee honestly — this is a heuristic with no
optimality claim; the optimality claims come in Section 7.

Checkpoint: total should drop to roughly **85–110**. `ghz_star` in particular should now be at or
near **6.5** (2 swaps), which is its proven optimum.

### Section 6 — `driver.py` — the portfolio

Purpose: the thing `solve()` actually calls. This is the module that makes the result reproducible
and the claims defensible.

Describe the loop in words:
1. Build the interaction graph. If a monomorphism exists, return that placement with a swap-free
   routing. Done.
2. Otherwise generate N placement candidates from all the seeds.
3. Route each one. Optionally route each in both the forward and reversed gate order if you
   implemented the reverse trick.
4. **Validate every candidate with your own validator** before it is eligible.
5. Return the best by your own `score`.
6. Wrap the whole thing in a wall-clock budget so it degrades gracefully instead of hanging.

Emphasise two design points for the presentation: the portfolio is exactly what Qiskit does
(`layout_trials`, `swap_trials`, `VF2Layout`, `VF2PostLayout`) — this is the industry-standard
architecture, not an improvisation. And selection is by *your own* metric, so nothing is fitted to
the grader.

Determinism: seed every RNG explicitly and say so in the writeup. A judge re-running the notebook
must get the same number.

### Section 7 — `exact.py` — A* (optional, high value for the slides)

Purpose: prove optimality rather than assert quality. This is what turns "our heuristic is good"
into "we match the proven minimum on four of six."

- State the scope honestly, because it is the single most important caveat: **A\* is an optimal
  router, not an optimal compiler.** It proves the minimum number of SWAPs *given a fixed initial
  placement*. On `ghz_star` it will prove 5 from a line seed and 7 from a spectral seed, while the
  true optimum over all placements is 2. Never call the number "the optimum" without the placement
  qualifier.
- State space: `(positions tuple, index of next unsatisfied gate)`. `g` = swaps used. Before
  expanding, advance the gate index past every gate that is already satisfied — this collapses huge
  parts of the tree.
- Two admissible heuristics to combine with `max(...)`:
  - **Distance bound**: over remaining gates, `max(distance(pos[a], pos[b]) - 1)`. At least that
    many swaps are needed for the worst remaining gate.
  - **Degree bound**: for each logical qubit, if it must interact with *p* distinct partners in the
    remainder and the chip's max degree is *d*, it needs at least `ceil((p - d) / 2)` swaps of its
    own; sum these and halve, since one swap serves two qubits.
  Both are lower bounds on the true remaining cost, so `max` of them is admissible and A* stays
  optimal. Walk the reader through *why* each is a lower bound — that argument is the presentation.
- Apply the same liveness prune as the router: skip any swap touching no live qubit.
- Cap the node count (400k is a reasonable figure) and fall back to the heuristic result on cap.

Expected behaviour to list as the checkpoint: `chain_trotter` and `vqe_layers` terminate instantly at
0 swaps; `ghz_star` proves 5 from the line seed in ~200 nodes; `ladder_trotter` proves 8 from a
spectral seed in ~39k nodes / ~3s; `qaoa_random` proves 10 from a spectral seed in ~209k nodes /
~14s; `dense_random` hits the cap on every seed. That last failure is the motivation for Section 8.

### Section 8 — `windows.py` — windowed exact routing

Purpose: the fix for A* capping out, and the good version of the "decompose into subcircuits" idea.

- Cut the gate list into **contiguous** windows of *k* gates. A*-solve window 1 starting from `P₀`
  and ending at whatever placement it lands in, `P₁`; solve window 2 starting from `P₁`; chain
  onward. Concatenate the routed segments.
- Because the windows are contiguous and solved in order, the concatenation preserves gate order by
  construction — it cannot violate `scorer.py:68`.
- *k* is a runtime/quality dial: larger *k* is closer to global optimal and exponentially slower.
  `dense_random` terminates on every window at k ≈ 8.
- The guarantee is real and stateable: **optimal within each window, given the placement the previous
  window ended in.** Say exactly that, and no more.
- Worth noting as an extension: letting the exact solver also choose the placement it *ends* in
  (rather than taking whatever falls out) is a better but much larger search; mention it as future
  work rather than building it.

Warn against the other reading of "decomposition": a **pattern library** keyed on recognising a
"star" or a "chain" and applying a stored solution. It will not generalise, and since the benchmarks
are literally named `ghz_star` and `chain_trotter`, a judge reads it as exactly the benchmark-fitting
you are trying to avoid.

### Section 9 — Tests

Two files, and the second one is the most valuable slide in the deck.

- **`tests/test_validity.py`** — every solver output on every benchmark passes `validate`; plus
  explicit regression tests for each trap in Section 1 (missing labels 4 and 7; pair orientation;
  caller's program unmutated; returned placement is the one actually used). Hint: the mutation test
  is the one that only fails on a cold run, so assert on a deep copy taken before the call.
- **`tests/test_generalization.py`** — run the solver on random programs against topologies it has
  never seen: `nx.random_regular_graph`, grids of several sizes, a heavy-hex at a different size, a
  path, a cycle. Assert validity every time, and record the ratio to the baseline. This is the
  answer to "you just fit the six benchmarks" — a chart over ~200 unseen instances ends the question
  in one slide.

### Section 10 — Wiring up and writing up

- Notebook **cell 20** becomes a thin wrapper: import `solve` from `solver`, call it. Note that the
  cell currently hardcodes `test_name = "ghz_star"`; the final version should loop all six and print
  the total.
- `writeup.md`, 1–2 pages: the pipeline diagram, the VF2 short-circuit, the portfolio, the two
  admissible bounds with their proofs in one sentence each, the generalisation result, and an honest
  limitations paragraph (A* is per-placement; windowed routing is per-window; `dense_random` is not
  proven optimal).
- Demo, 3–5 minutes: `starter_kit/visualize.py` already provides `animate_routing` and
  `animate_layers`. Use the baseline-versus-yours animation on `ghz_star` — it is the most legible
  visual, 7 swaps down to 2.

### Closing: score milestones

A table the reader can check themselves against:

| After stage | Expected total | What changed |
|---|---|---|
| baseline | 283.5 | — |
| §3 embed | ~244 | two benchmarks solved exactly |
| §4 router | ~150–200 | valid routing, naive placement |
| §5 placement | ~85–110 | the main lever |
| §6 portfolio | ~75–85 | best-of-N selection |
| §7–8 exact | ~70–80 | plus optimality claims |

Floor if every benchmark were swap-free: **24.0** (unreachable — four of six are provably not
embeddable).

---

## Verification for this task

1. `Computational Track/plan.md` exists and contains all ten sections plus the milestone table.
2. Every file path, line reference, and API name in it resolves: `scorer.py:15`, `:41-43`, `:50`,
   `:68`, `:78-79`; `baseline_routing.py:45`; `benchmarks.py:50` for the `range(12)` trap;
   `starter_kit/visualize.py` exports `animate_routing`, `animate_layers`, `draw_hardware`.
3. The document contains **no implementation code** — file trees, expected console output, and
   interface descriptions only.
4. Read it end to end as if you had not seen the prior analysis, and confirm each stage's checkpoint
   is something you could actually run and compare a number against.
