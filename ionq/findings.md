# Quantum Duel — Cup 1 findings

**Ruleset:** `quantum-duel-4q-playtest-0.3` (4 qubits). Platform RC 0.7.2.
**Status:** local practice measurements only. Nothing here is an official score, and
no result was obtained against a real opponent.
**Scripts:** `ionq/dev/experiments/`. Harness: `ionq/dev/harness.py`.

Sample sizes are small (2–3 draw seeds per attack family). Treat point values as
indicative of large effects, not as precise estimates. Everything marked
*measured* was run; everything marked *derived* follows from the SDK source.

---

## 1. The contract (derived from `qduel_sdk/rules.py`)

| Quantity | Value |
|---|---|
| Qubits | 4 (dimension 16) |
| Total shots | 12,000 |
| Checkpoints | 3, so a **block is exactly 4,000 shots** |
| Distinct settings | 180 for the whole encounter |
| Actions (`max_requests`) | 600 for the whole encounter |
| Attack circuit | ≤ 12 gates, ≤ 4 entanglers |
| Patch circuit | ≤ 18 gates, ≤ 6 entanglers |
| Gate alphabet | `rx ry rz rxx ryy rzz`, angles in [−π, π] |
| Scoring | ε = 1 − \|Tr(PA)\|²/256; 100 pts at ε ≤ 0.001, 0 pts at ε ≥ 0.1, log-linear between |
| Encounter recovery | mean of the three checkpoint scores |
| Match score | (your recovery + (100 − their recovery)) / 2 |
| Minimum attack disturbance | every accepted draw must have ε(A) ≥ 0.1 |

A "setting" is one (preparation, analysis circuit, readout basis) triple; changing
the analysis circuit creates a new setting. A batch of 32 queries consumes 32
actions, not one (`LocalSession._save` increments per item) — batching saves
network round trips, not budget.

### 1.1 Shot allocation across checkpoints is not a decision

`available_now = (stage+1)·block − spent − forfeited`. You cannot spend more than
4,000 before closing checkpoint 1, and closing early forfeits the remainder
permanently. The allocation is therefore **forced to 4000 / 4000 / 4000**. The
only genuine allocation freedom is across settings *within* a block.

### 1.2 Angle error budget

For a residual error generator with total angle deviation δ on a single Pauli
rotation, ε = sin²(δ/2) ≈ δ²/4. So:

- 100 points (ε ≤ 0.001) needs δ ≲ 0.063 rad
- 0 points (ε ≥ 0.1) starts at δ ≈ 0.64 rad

With k residual generators, ε ≈ ¼·δᵀCδ where C is the Gram matrix of the
transported Pauli generators.

---

## 2. The central result

**This game is architecture identification under a compactness constraint.
Measurement design and shot allocation are second-order.**

Three independent measurements support this.

### 2.1 The information is always sufficient (*measured*, `oracle_arch.py`)

Give the defender the true gate architecture and let it fit only the angles from
the same 180-setting panel:

| attack | seed | N | stock learner | arch known, warm start | arch known, 12 random starts |
|---|---|---|---|---|---|
| `near_clifford` | 41 | 4000 | 0.0 | **91.4** | 0.0 (0/12) |
| `near_clifford` | 41 | 12000 | 0.0 | **100.0** | 0.0 (0/12) |
| `near_clifford` | 73 | 4000 | 0.0 | **82.5** | 0.0 (0/12) |
| `near_clifford` | 73 | 12000 | 0.0 | **100.0** | 12.3 (0/12) |
| `public_ladder` | 41 | 12000 | 0.0 | **100.0** | 0.0 (0/12) |
| `public_ladder` | 73 | 12000 | 27.3 | **100.0** | 0.0 (0/12) |
| `interleaved_cycle` | 73 | 12000 | 100.0 | **100.0** | 100.0 (1/12) |
| `star` | 73 | 12000 | 67.6 | **100.0** | 0.0 (0/12) |
| `scrambler` | 41 | 12000 | 100.0 | **100.0** | 100.0 (1/12) |

`archTruth` reaches 100.0 at N = 12,000 on **every** attack tested. The panel
always carries enough information. Every observed failure is a search failure.

`archRand` shows the second half: even *with the architecture handed over*,
random-start optimisation of 12 angles finds the right basin roughly 0–8% of the
time. Likelihood correctly identifies the right basin whenever a restart lands in
it, so restarts work in principle — the basin is just small.

### 2.2 The Cramér–Rao floor is never binding (*measured*, `crb.py`)

Floor on ε assuming the architecture is known, for several measurement designs:

| attack (params) | baseline panel N=4000 | N=12000 | closed loop (analysis = inverse) N=4000 |
|---|---|---|---|
| `zz` (3) | 2.89e-4 → 100 pts | 9.64e-5 | 9.20e-4 → 100 |
| `conjugated` (6) | 8.18e-4 → 100 | 2.73e-4 | 1.78e-3 → 87 |
| `commutator` (6) | 1.54e-3 → 91 | 5.12e-4 | 2.20e-3 → 83 |
| `mixed` (8) | 1.16e-3 → 97 | 3.88e-4 | 2.20e-3 → 83 |
| `ladder` (12) | 2.05e-3 → 84 | 6.83e-4 | 3.31e-3 → 74 |

Two consequences:

1. **Feeding your current patch back as the analysis circuit makes things worse**,
   by roughly 2×, on every family. The prep-matched "echo" variant is
   catastrophic: the first-order Fisher information vanishes identically at the
   null (measured F eigenvalues ~1e-21), because a deterministic outcome has zero
   first-order sensitivity. The intuitive "measure the residual" design is a trap.
2. **ε scales at roughly 1.7e-4 per parameter at N = 4,000.** Since 12 gates is
   the hard cap, the worst a purely statistical attack can do is ε ≈ 2.0e-3 at
   checkpoint 1 → **84 points**, and ~95 for the encounter average. *There is no
   statistics-only attack in this ruleset.*

### 2.2.1 What optimal shot allocation is worth (*measured*, `alloc.py`)

A-optimal here is an **oracle** design that already knows the attack, so it is an
unreachable upper bound on any adaptive scheme:

| attack | N | uniform 180 | A-optimal | unif 60 | unif 90 | pts uniform | pts A-opt |
|---|---|---|---|---|---|---|---|
| `mixed` | 4000 | 1.164e-3 | 5.61e-4 | 1.19e-3 | 1.13e-3 | 96.7 | 100.0 |
| `ladder` | 4000 | 2.048e-3 | 1.52e-3 | 2.02e-3 | 2.09e-3 | 84.4 | **90.9** |
| `conjugated` | 4000 | 8.18e-4 | 3.75e-4 | 8.99e-4 | 8.45e-4 | 100.0 | 100.0 |
| all three | 12000 | — | — | — | — | 100.0 | 100.0 |

**Perfect allocation buys at most ~6.5 points, at checkpoint 1 only, and nothing
by checkpoint 3** — roughly 2 points on the encounter average, against a search
gap of 0 → 100.

Two further details. The optimal design concentrates onto **5–24 of 180 settings
at up to 58× uniform weight**, so a realizable scheme would have to guess that
concentration without knowing the attack, and guessing wrong costs identifiability.
And `unif60 ≈ unif90 ≈ unif180` to three digits: the *number* of settings barely
affects variance. Keep 180 for search robustness, not for precision.

### 2.3 The patch cap forces architecture recovery (*measured*, `ansatz.py`)

Can a fixed 18-gate / 6-entangler circuit shape invert an arbitrary legal attack,
letting the defender skip identification entirely? Tested against the **exact**
attack unitary with zero shot noise — an upper bound on any defender:

```
trial   best fixed shape   best over 27 axis assignments
    0          2.864e-01                      1.916e-01
    1          1.704e-01                      1.562e-01
    2          7.684e-01                      4.184e-01
    3          7.347e-01                      1.309e-01
    4          7.084e-01                      4.292e-01

true inverse architecture                      0.000e+00
generic 18-gate / 6-entangler ansatz           6.346e-01
```

Every fixed shape lands between 0.13 and 0.43, all above the 0.1 zero-credit
threshold. The patch budget is only 1.5× the attack budget, which is too tight to
absorb a wrong architecture. **The defender must recover essentially the
opponent's own circuit structure**, and the 6 spare gates cannot buy it out.

---

## 3. Refuted hypotheses

Recording these because they were plausible and are wrong.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Scrambling the state makes recovery hard | **Backwards.** `scrambler` (broad angles on all 12 gates, maximum output entropy) was the *easiest* attack tested at 90.0 mean recovery. More scrambling means more distinguishable counts. | `screen.py` |
| Closed-loop residual measurement (analysis = current patch) helps | **Worse by ~2×.** Prep-matched echo is far worse still. | `crb.py` |
| Effective weight-3 generators escape the learner's weight-≤2 insertion dictionary | **No effect.** `weight3_nest` scored 84.3. | `screen.py` |
| A fixed universal ansatz lets the defender skip identification | **Impossible** at 18/6. | `ansatz.py` |
| The stock learner's warm-starting is uniformly harmful across checkpoints | **Bidirectional, not uniform.** See §5.1. | `oracle_arch.py` vs `screen.py` |
| Randomized restarts on gate pursuit beat the stock single pass | **Lost everywhere** — the test omitted the dense relaxation, so it measured the wrong thing. See §5.2. | `strong.py` |
| Restarts around dense→project beat stock | **Not uniformly** — big wins on `ladder`, but lost a 100.0 on `interleaved_cycle` by discarding stock's other candidate generators. See §5.3. | `strong2.py` |
| Seeding the search at quarter turns cracks `near_clifford` | **No effect** — 0.0 at all six configurations. See §5.4. | `cliff.py` |
| `near_clifford` works by stopping greedy search early | **No** — the pursuit runs to the 18-gate cap, first gain 5000× the BIC penalty. It is a decoy optimum. See §5.4. | `diag.py` |
| Merged insertions are the right hedge against a frame-aware defender | **Backwards for Cup 1** — the exact quarter turns they require are what makes the frame learner win (100/94/100). Ranged angles beat both learners. See §5.6. | `frame4.py` |
| The parity panel lets both generators share counts for free | **No** — it cost 100→0 on `conjugated` and `zz` by gutting panel diversity. See §5.7. | reverted |
| Maximising parameter count is a viable attack | **Capped at ~84 pts of damage.** | §2.2 |

---

## 4. Attack archetype screen (*measured*, `screen.py`)

Eight candidate 12-gate / 4-entangler templates, three draw seeds each, against the
stock SDK defender. Lower recovery = stronger attack. All passed admission
readiness 64/64.

| archetype | mean recovery | best case for defender | mechanism |
|---|---|---|---|
| **`near_clifford`** | **0.0** | 0.1 | uniqueness denial (Clifford degeneracy) |
| `public_ladder` | 17.1 | 44.3 | uniqueness + precision |
| `star` | 31.9 | 55.5 | precision (support overlap) |
| `interleaved_cycle` | 36.9 | 86.8 | precision (non-commuting interleave) |
| `shared_chain` | 80.0 | 97.2 | precision (parameter correlation) |
| `repeated_pair_sandwich` | 83.8 | 86.8 | precision (confounded angles) |
| `weight3_nest` | 84.3 | 92.4 | model-class escape — refuted |
| `scrambler` | 90.0 | 94.4 | entropy — refuted |

`near_clifford`: 12 gates / 4 entanglers, eight slots with ranges [1.45, 1.69]
straddling π/2, four off-Clifford locals in [0.8, 1.3]. It produced ε between 0.37
and 0.83 at **every checkpoint of every seed**, and is the only archetype with no
seed on which the defender recovered anything.

### 4.1 Mechanism: axis-permuting quarter-turns (*measured*, `tune.py`)

Cluster point and half-width varied, 12 gates / 4 entanglers, 3 seeds each:

| variant | mean recovery | s41 | s73 | s907 |
|---|---|---|---|---|
| cluster at π/2, half-width 0.12 | **0.0** | 0.0 | 0.0 | 0.0 |
| cluster at π/2, half-width 0.02 | **0.0** | 0.0 | 0.0 | 0.0 |
| cluster at π/2, half-width 0.40 | **0.0** | 0.0 | 0.0 | 0.0 |
| cluster at π, half-width 0.12 | 87.1 | 81.4 | 90.2 | 89.6 |
| cluster at 1.0 rad (non-Clifford) | 84.1 | 81.9 | 84.1 | 86.4 |
| cluster at 2.2 rad (non-Clifford) | 89.6 | 90.0 | 87.1 | 91.8 |

Three conclusions:

1. **Not "narrow ranges."** Half-width 0.40 — a range of [1.17, 1.97] — traps just
   as completely as 0.02.
2. **Non-Clifford cluster points do not trap.** The falsifier passed.
3. **Clustering at ±π does not trap either**, which refines the mechanism.
   `R_P(π) = −iP` is Clifford but is a *Pauli*: conjugating by it only flips
   signs. `R_P(π/2)` conjugation genuinely **permutes** Pauli axes (X→Y). The
   degeneracy requires **axis-permuting** conjugation, so quarter-turns
   specifically.

Since `C·R_P(θ) = R_{CPC†}(θ)·C`, a circuit of near-quarter-turns admits a large
family of nearly-equivalent reorderings — a group-theoretic degeneracy, not an
artefact of one learner's search. Corroboration: Cup 2's frame grammar is built
from ±π/2 quarter-turns (`validate_frame_template` checks
`abs(abs(lo) − π/2) < 1e-12`), and the organizers ship `merged_frame_example` as a
trap *within* that structure, so they treat quarter-turn frames as the hard case
too.

---

## 5. Stock defender behaviour

Baseline (`duelkit.recovery4.adapter.run_defender`, `grouped=True`) against the
public bank, seeds 41 and 73:

```
4q_commutator@41  100.00   4q_commutator@73  100.00
4q_conjugated@41  100.00   4q_conjugated@73  100.00
4q_ladder@41        0.00   4q_ladder@73       44.32
4q_local@41       100.00   4q_local@73       100.00
4q_mixed@41        80.42   4q_mixed@73        92.21
4q_zz@41          100.00   4q_zz@73          100.00
4q_warmup_rzz     100.00
mean 85.92   worst 0.00
```

Failures are **bimodal**: either ε ≈ 1e-4 (structure found, 100 pts) or ε ≈ 0.15+
(structure missed, 0 pts). There is almost nothing in between, which is what a
search-limited rather than noise-limited problem looks like.

**Runtime: 2.6–4.0 s single-threaded per full encounter**, against a documented
local smoke timeout of 180 s. Roughly 45× compute headroom is unused. (The server
runtime limit is not published; 180 s is the local default in
`submission.smoke_submission`.)

### 5.1 The uncontrolled-trajectory defect

`GatePursuit.fit` starts from `self.labels` and only ever *inserts* gates — it
cannot delete, reorder, or change an axis. `DenseMLE.fit` warm-starts from
`self.x`. `Ensemble` holds both across all three checkpoints. Comparing the
warm-started encounter curve against a fresh fit on the same total data:

| case | warm-started CP3 | fresh fit, same N |
|---|---|---|
| `interleaved_cycle@73` | **0.2** | **100.0** |
| `star@73` | **4.2** | **67.6** |
| `public_ladder@73` | **100.0** | **27.3** |
| `scrambler@41` | 100.0 | 100.0 |

Warm-starting rescued one case and destroyed two others. The defect is not that
warm-starting is bad — it is that the learner commits to one trajectory with **no
mechanism to compare it against the alternative**. Running both and selecting by
likelihood takes the max of the two columns at negligible cost.

---

### 5.2 The dense relaxation is load-bearing (*measured*, `strong.py`)

A randomized-restart searcher that ran greedy gate pursuit **directly on the count
likelihood**, ~230 restarts per checkpoint, 40 s budget:

| attack | seed | N | stock | restart searcher | restarts |
|---|---|---|---|---|---|
| `near_clifford` | 41 | 12000 | 0.0 | 0.0 | 225 |
| `public_ladder` | 73 | 12000 | 27.3 | **0.0** | 275 |
| `interleaved_cycle` | 73 | 12000 | **100.0** | **0.0** | 247 |
| `star` | 73 | 12000 | **67.6** | **0.0** | 230 |

It lost everywhere, including cases stock solves perfectly. The difference is that
stock runs `DenseMLE` → unconstrained 16×16 target → `GatePursuit` against
`MatrixDistance(target)` → refine on counts, while this searcher skipped the dense
relaxation. **One pass of dense-relax-then-project beats 230 passes of direct
likelihood search.** The two-stage structure matters far more than the restart
count.

### 5.3 Extend the candidate pool; never replace it (*measured*, `strong2.py`)

Restarts wrapped *around* dense→project, plus delete/swap/axis local-search moves
that stock's insert-only pursuit cannot make, 40 s per checkpoint:

| attack | seed | N | stock | strong2 | iters | ε strong2 |
|---|---|---|---|---|---|---|
| `near_clifford` | 41 | 12000 | 0.0 | **64.9** | 24 | 5.03e-3 |
| `near_clifford` | 73 | 12000 | 0.0 | 0.0 | 15 | 3.60e-1 |
| `public_ladder` | 73 | 4000 | 16.3 | **64.2** | 48 | 5.20e-3 |
| `public_ladder` | 73 | 12000 | 27.3 | **100.0** | 42 | 8.36e-4 |
| `interleaved_cycle` | 41 | 12000 | 52.4 | **0.0** | 12 | 6.43e-1 |
| `interleaved_cycle` | 73 | 12000 | 100.0 | **0.0** | 18 | 2.13e-1 |
| `star` | 41 | 4000 | 47.5 | 49.7 | 73 | 1.01e-2 |
| `star` | 73 | 12000 | 67.6 | 67.6 | 28 | 4.44e-3 |

Big wins on `ladder@73` and the first partial crack of `near_clifford` — but it
destroyed `interleaved_cycle`, turning a 100.0 into 0.0.

Cause: like `strong.py`, this replaced rather than extended. Stock's `Ensemble`
generates **three heterogeneous candidates** — `direct` (pursuit on the
likelihood), `projection` (pursuit against the dense target) and `block` (the
shell-conjugation motif fit) — and selects among them by BIC. `strong2` kept only
the projection path, discarding the two generators that happened to work on
`interleaved_cycle`. Low iteration counts there (12–18 vs 42–73) show the local
search also consumed the budget.

**Design rule, established by two independent failures:** stock's strength is the
heterogeneous candidate pool under BIC selection, not any single search path. A
stronger defender must **strictly extend** that pool — keep `direct`,
`projection`, `block`; add restarted dense targets, local-search-refined variants,
and a from-scratch fit alongside the warm one; select across the whole enlarged
pool. That construction cannot score below stock except through selection noise,
and §2.1 showed likelihood reliably identifies the right basin when a candidate
lands in it.

Row-wise `max(stock, strong2)`, which pool-union approximates: `ladder@73` 100.0,
`near_clifford@41` 64.9, `interleaved_cycle@73` 100.0, `star` unchanged.

**`near_clifford` is not strictly unbreakable.** One of four configurations
reached 64.9. It remains by far the strongest attack measured, but the claim
should be "0.0 against the stock learner at every seed tested, and 3 of 4 against
an extended searcher", not "unrecoverable".

### 5.4 The trap is a decoy optimum, not a search-budget problem (*measured*, `cliff.py`, `diag.py`)

Two explanations for `near_clifford` were tested and both are wrong.

**"The search never looks where the answer is."** Stock inserts each new gate at
angle ~0.08 while the true angles sit at ±π/2, so insertions were seeded at
{+π/2, −π/2, small} and a snap-to-quarter-turn move added (`cliff.py`):

| attack | seed | N | stock | Clifford-seeded |
|---|---|---|---|---|
| `near_clifford` | 41 | 12000 | 0.0 | 0.0 (ε 8.52e-1) |
| `near_clifford` | 73 | 12000 | 0.0 | 0.0 (ε 9.70e-1) |
| `near_clifford` | 907 | 12000 | 0.0 | 0.0 (ε 9.67e-1) |
| `interleaved_cycle` | 41 | 12000 | 52.4 | 100.0 |
| `scrambler` | 907 | 12000 | 100.0 | 59.8 |

No effect on the target, non-monotone elsewhere. Refuted.

**"No single insertion is BIC-significant, so greedy halts."** Also wrong
(`diag.py`):

| attack | seed | true gates | direct gates | proj gates | first-insertion gain | BIC penalty |
|---|---|---|---|---|---|---|
| `near_clifford` | 41 | 12 | 18 | 18 | 46646.9 | 9.39 |
| `near_clifford` | 73 | 12 | 18 | 18 | 41392.8 | 9.39 |
| `public_ladder` | 41 | 12 | 18 | 18 | 50242.5 | 9.39 |

The pursuit runs to the full 18-gate cap and the first gain exceeds the penalty by
~5000×. Nothing stops early and the landscape is not flat.

**What is actually happening.** The searcher spends its entire budget on a *wrong
18-gate structure that fits the counts well*. The true model fits better
(`archTruth` reaches ε ≈ 1e-4), but the near-Clifford circuit admits deep
competing optima in the compact-model class and greedy descent reaches one first.
This is branch B of §MECE — **a decoy global optimum** — and it explains why four
separate searchers failed: restarts, dense+restarts+local search, Clifford
seeding, and random multistart with the architecture given away. It is not a
compute problem, so compute does not fix it.

### 5.5 Topology-first identification works (*measured*, `topo.py`)

Coupling read off the PTM of the dense estimate: for each pair (i,j), the weight
that U P_i U† places on operators with support on j. Top-k pairs against the
attack's true entangler pairs, N = 12,000:

| attack | seed | true pairs | top-k by coupling | hit |
|---|---|---|---|---|
| `near_clifford` | 41 | (0,1)(0,3)(1,2)(2,3) | (2,3)(1,2)(0,1)(0,3) | 4/4 |
| `near_clifford` | 73 | same | (0,1)(1,2)(0,3)(2,3) | 4/4 |
| `public_ladder` | 41 | same | (2,3)(0,1)(0,3)(0,2) | 3/4 |
| `star` | 41 | (0,1)(0,2)(0,3) | (0,2)(0,3)(0,1) | 3/3 |
| `interleaved_cycle` | 73 | (0,1)(0,3)(1,2)(2,3) | (2,3)(0,1)(0,3)(1,2) | 4/4 |
| `scrambler` | 73 | same | (0,1)(0,3)(2,3)(1,3) | 3/4 |

3/4 or better in all ten cases, 4/4 in six. **Entangler support is identifiable
even when the full architecture is not**, which shrinks the entangler dictionary
from 18 types (6 pairs × 3 axes) to ~12. Whether that improves recovery is
untested; since §5.4 shows `near_clifford` fails through decoy optima rather than
search-space size, expect gains elsewhere rather than there. This is the strongest
untried lever.


### 5.6 Porting the recovery8 frame learner to 4 qubits (*measured*, `frame4.py`)

`recovery8/quantum.py` is dimension-general and `Recovery` takes `n` as a
parameter, so the only hard 8 is the adapter's guard. Ported with `n = 4`:

| attack | seed | stock | pooled | frame4 | secs |
|---|---|---|---|---|---|
| Exactly-Clifford frame w/ merged insertions | 41 | 0.0 | 0.0 | **100.0** | 0.2 |
| Exactly-Clifford frame w/ merged insertions | 73 | 0.0 | 28.4 | **94.2** | 0.2 |
| Exactly-Clifford frame w/ merged insertions | 907 | 0.0 | 0.0 | **100.0** | 0.2 |
| Ranged quarter-turn frame | 41/73/907 | 0.0 | 27/64/0 | **0.0** | 0.1 |
| `4q_ladder` | 41 | 0.0 | 56.5 | 0.0 | 0.1 |
| `4q_conjugated` | 41 | 100.0 | 100.0 | 0.0 | 76.7 |
| `4q_zz` | 41 | 100.0 | 100.0 | 66.7 | 3.9 |

**The decisive distinction is exact vs ranged quarter turns.** A frame whose gates
sit at *exactly* ±π/2 has surviving Pauli relations that `ParityLearner` recovers
exactly, reaching full rank and synthesising the frame in 0.2 s. A frame with
angles *ranged* across π/2 (e.g. [1.45, 1.69]) is never exactly Clifford, the
relations do not hold, and the algebraic route returns nothing.

Consequence for the attack: the Cup 2 `merged_frame_example` countermeasure is
**wrong for Cup 1**. Cup 2's grammar forces exact quarter turns, so merging
insertions is the only available deception there. Cup 1 has no such constraint,
so ranged near-quarter-turns are strictly better — they defeat the stock learner
*and* the frame learner. Both submitted templates use ranged angles.

Frame4 also takes 76.7 s on `4q_conjugated`, so any integration needs a time guard.

### 5.7 Sharing the panel with the frame learner fails (*measured*)

Integrating the frame candidate into the pool requires settings. To avoid a
budget conflict the 180-setting panel was rebuilt as 36 parity-compatible
families of (base + 4 single-qubit-flipped preps) in a shared readout basis, so
both generators could read the same counts. Result:

| attack | before | after |
|---|---|---|
| `4q_conjugated@41` | 100.0 | **0.0** |
| `4q_zz@41` | 100.0 | **0.0** |

`make_panel` samples all six preps independently per qubit; the parity panel
restricts base preps to {+, +i, 0} and makes four of every five settings differ
from the base in a single qubit within one readout basis. That is far less
informationally diverse, the dense MLE degrades, and easy attacks stop being
recoverable. **Reverted.** The frame candidate is not shipped; a dedicated
settings slice remains untested and would take budget from a panel that §2.2.1
shows is already near-optimal.


---

## 6. Open questions

1. **Does restricting the search to the identified topology (§5.5) improve
   recovery?** Support detection works; its downstream value is unmeasured.
   Highest-value next experiment.
2. **Is there a non-greedy route past the decoy optima of §5.4?** Joint
   multi-gate moves, or a global search over entangler skeletons within the
   identified topology. Everything greedy has failed.
3. Whether the server's runtime limit is anywhere near the local 180 s default.
   The defender targets 75 s to stay well inside it.

---

## 7. Recommended strategies

### 7.1 Defence

Order of leverage, measured: **architecture search ≫ checkpoint policy ≫ shot
allocation ≫ observable choice.**

**Measurement layer — leave it alone.** Keep the 180-setting random product panel,
re-queried each block with counts accumulated, 4,000 shots per block split evenly
(22 + 1 for the first 40). Budget use: 540 queries + 3 patches + 3 closes = 546 of
600 actions. Do not build adaptive allocation (≤ ~2 encounter points, §2.2.1), do
not use closed-loop analysis circuits (actively worse, §2.2), do not reduce the
setting count (no variance benefit, and settings aid identifiability).

**Estimation layer — everything goes here.** Build a **strict superset** of the
stock candidate pool, per §5.3:

- keep all three stock generators (`direct`, `projection`, `block`)
- add a **from-scratch** fit at each checkpoint alongside the warm continuation
  (§5.1: warm-starting is bidirectional; selection takes the max for free)
- add restarted `DenseMLE` targets, each projected separately
- add **delete / swap / axis-change** local search — stock's pursuit is
  insert-only and can never revise a wrong gate
- select across the whole pool by BIC on the accumulated counts, then refine the
  winner on counts

**Measured result** (`main.py`, 13 public cases at seeds 41/73 plus both own
templates, `TIME_BUDGET = 55 s`, stop after 10 restarts without improvement):

| | stock | pooled defender |
|---|---|---|
| mean (public bank) | 85.9 | **92.1** |
| worst (public bank) | 0.0 | **56.5** |
| failures | 0 | 0 |
| regressions | — | 1 of 17 (−0.7, selection noise) |
| max runtime | 4.0 s | 55.0 s (37.4 s on the public bank) |

Biggest single gain: `4q_ladder@41` 0.0 → 56.5. The zero-point case is
eliminated. Max runtime is 31% of the documented 180 s local default, so the
defender survives grading hardware roughly 3x slower than an Apple M2. Hard
attacks reach the 55 s cap; easy ones finish in 5-8 s via the staleness exit.
The server limit is not published, so this margin is an assumption, not a fact.

Two bugs found while building this, both worth remembering: `min(pool)` over
`(deviance, model)` tuples raises `TypeError` on ties because it falls through to
comparing lists of dicts, and it crashed on exactly the *easy* attacks where
candidates converge to identical fits; and `DenseMLE.fit` seeds a **fixed** RNG
(`default_rng(83177)`), so a second `Ensemble` reproduces the first exactly —
diversified restarts require perturbing `dense.x` by hand.

**Robustness layer — cheap and mandatory.** Submit a legal patch at *every*
checkpoint, never leave it empty; validate before submitting (a rejected patch
silently keeps the previous one); wrap everything so no exception escapes
(`DEFENDER_EXCEPTION` forfeits the encounter); track `available_now` and the
setting count so no limit is tripped mid-run.

### 7.2 Attack

Only the uniqueness-denial branch works. Precision-denial is capped at ~84 points
of damage (§2.2), expressibility-denial is provably empty (`instantiate` verifies
a legal inverse for every accepted draw), and entropy is counterproductive (§3).

**Both slots quarter-turn frames, structurally different from each other:**

Both slots are **ranged** near-quarter-turn frames — every angle straddles ±π/2
but is never exactly at it. §5.6 is why: exact quarter turns fall to the ported
frame learner in 0.2 s, ranged ones defeat it entirely.

- **Slot 1** — *cycle* topology: entanglers on (0,1), (2,3), (1,2), (0,3).
- **Slot 2** — *crossed* topology: entanglers on (0,2), (1,3), (0,1), (2,3),
  different local axes.

Each is 12 gates / 4 entanglers: eight slots ranged in [1.45, 1.69] or
[−1.69, −1.45], four off-Clifford locals in [0.8, 1.3]. Both pass `qualify`
(readiness 64/64, VALIDATED).

**Measured**, four seeds each:

| template | vs stock | vs ported frame learner | vs pooled defender |
|---|---|---|---|
| cycle | **0.0** | **0.0** | 34.7 |
| crossed | **0.0** | **0.0** | 11.0 |

`crossed` is the stronger of the two. The residual exposure is a pooled defender,
which is strictly stronger than either individual learner.

Both templates must pass `qualify`: distinct fingerprints, distinct
`semantic_template_key`, ≥32/64 readiness draws, and not equivalent on the eight
paired validation draws.

### 7.3 Build order

Attack corpus first, stratified **by mechanism** so the defender does not overfit
to quarter-turn frames:

| stratum | families | purpose |
|---|---|---|
| quarter-turn degeneracy | plain frame, merged-insertion frame, varied widths and topologies | the only working attack mechanism |
| precision denial | `ladder`, `star`, `interleaved_cycle` | where extended search should pay |
| easy controls | `local`, `zz`, `conjugated`, `commutator` | regression guard |
| refuted but legal | `scrambler`, `weight3_nest` | opponents will submit these |

Fixed seeds per family across defender revisions; fresh seeds reserved for the
final check.
