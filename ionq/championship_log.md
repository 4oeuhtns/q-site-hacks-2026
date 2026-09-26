# Quantum Duel Championship (8-qubit open final) — work log

Deadline: Friday 2026-09-25, 11:59 PM. Log started Thursday 2026-09-24.
Ruleset: `quantum-duel-8q-open-0.7.1`. Everything below is **local practice
measurement**; nothing is an official score.

**Strategy summary: see §11.**

Raw data: `dev/open8/data/results.jsonl` (201 full encounters so far, one JSON row each:
spec, draw, per-checkpoint points and infidelity, CPU seconds, defender log tail).
Aggregate with `python3 dev/open8/summarize.py [tag ...]`.

---

## 1. The contract

| Quantity | Value |
|---|---|
| Qubits | 8 (dimension 256; judge denominator 65,536) |
| Shots | 96,000 = 3 checkpoints × 32,000 (use-or-lose per block) |
| Distinct settings / actions | 720 / 3,000 |
| Attack | 1–72 rotations, ≤ 24 entanglers, any order, any pairs |
| Patch | ≤ 108 gates, ≤ 36 entanglers |
| Gates | `rx ry rz rxx ryy rzz`, angles in [−π, π] |
| Scoring | ε = 1 − \|Tr(PA)\|²/4ⁿ; 100 pts at ε ≤ 0.001, 0 at ε ≥ 0.1, log-linear between |
| Match | (our recovery + (100 − their recovery)) / 2 |
| Championship sampling | 3 draws × 2 repeats per pairing (notebook cell 64) |

The notebook says plainly that unrecoverable "scrambler" templates are legal and
that "ranks separate on the attacks that can be recovered".

## 2. Starting point: Cup 2 official feedback

File: `quantum_duel_work/quantum-duel-8q-frame2-0.7/team-feedback-…json`.

| | Result |
|---|---|
| Defence (f2def) | **100.00** on all 48 encounters vs 12 opponents; median ε 8.8e-6, max 1.3e-4 |
| Attack | **91.67**: 11 of 12 teams recovered 0; team `9d05da…` recovered **both** slots at 100 |
| Failures | 0 |

Reading: the field is one strong rival plus eleven teams that could not defend a
72-gate Clifford frame. Our Cup 2 attacks were exact quarter-turn frames with two
non-Clifford local insertions ("Compile-hard frame A/B").

## 3. Planning (bias-countered brainstorm)

Methods used: Assumption Surfacing, First Principles, MECE, Perspective Forcing,
Inversion. Key conclusions before any code:

1. **Cup 1 lessons may invert at 8q.** Cup 1 (4q) found "scrambling is the easiest
   attack", but that relied on a dense 16×16 unitary fit. At 8q the dense fit is
   256×256 (~65k parameters vs 96k shots), so scramblers should become strong.
2. **Defence points come from recoverable opponent attacks.** Unrecoverable
   attacks tie every defender. The winning defender must recover every attack that
   *can* be recovered.
3. **Known-architecture prior is the cheapest gain.** Cup 1 §2.1 showed that with the
   architecture known, angle fitting reached 100 on every attack. Teams that copy
   public templates publish their architecture and angle centres.
4. **Inversion checklist:** defender crash on OPEN8 = forfeit everything; timeouts;
   ZIP built against the wrong ruleset; attacks failing `qualify`.

Discovered while reading the notebook (cell 74): **the notebook's default final
attacks are the public `mixed` (open_example seed 31, 18 gates / 6 ent) and
`multilayer` (seed 37, 72 gates / 24 ent) templates.** Any team that does not edit
the defaults submits exactly these.

## 4. Tooling built

All under `ionq/dev/open8/` (dev only) and the solution directory
`ionq/quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/` (submitted).

| File | Purpose |
|---|---|
| `dev/open8/lab.py` | Attack families + full `LocalSession` encounter runner, process pool, JSONL logging |
| `dev/open8/offline.py`, `t_pursuit.py` | Fast offline loop: random 360-setting panel → 32k shots → one learner → ε vs truth |
| `dev/open8/summarize.py` | Aggregates results by tag and family |
| `dev/open8/build_tlib.py` | Generates `tlib.json`, the public-architecture library (533 entries) |
| `my_solution/sim8.py` | Batched 8q rotation simulator + adjoint count-likelihood gradient (NumPy and Numba paths) |
| `my_solution/pursuit.py` | Greedy gate pursuit for unknown architectures |
| `my_solution/opendef.py`, `main_open.py` | Pooled defender (see §6) |
| `my_solution/f2def.py`, `main.py` | Cup 2 defender, extracted unchanged from the Cup 2 ZIP |

Attack families in `lab.py`:
- `frame(k, width, pair_ins, layers)`: Cup-2-style layers (8 locals + 4 matching pairs) with k non-Clifford local insertions, optional near-Clifford ranging (±width around ±π/2), optional non-Clifford pair slots.
- `generic(gates, ents, bands, pairs)`: random continuous-angle architecture; `pairs` ∈ random / brick / all; `bands` restricts angle magnitudes (used [0.6,1.2] ∪ [1.9,2.5] to avoid 0, ±π/2, π).
- `sparse(gates, ents)`: short generic circuits.
- `open_example`, `pub` (practice bank), `frame_example`, `merged`, `cup2_own`, `file`.

### Simulator verification
- Probabilities match the SDK's `stream_probability` to 1.9e-16.
- Adjoint gradient matches finite differences (22.0716 vs 22.0724 at h = 1e-6).
- Numba path matches NumPy to 3e-12; about 2.7× faster under load.

## 5. Experiment 1 — attack sweep vs f2def (`sweep1_f2def`, 118 encounters, 0 crashes)

f2def ran on OPEN8 unchanged (its only guard is `qubits == 8`).

| Family | n | mean recovery | CPU s |
|---|---|---|---|
| frame k=0 (pure Clifford) | 4 | 100 | 6 |
| frame k=1 / k=2 | 4 / 4 | 100 / 100 | 33 / 50 |
| frame k=2, layers 1/2/3 | 2 each | 100 | ~50 |
| frame k=4 | 4 | 100 | **514** |
| frame k=6, 8, 12, 16, 24 | 4 each | **0** | ~17 |
| frame k=2, width 0.05 | 4 | 29.6 | 182 |
| frame k=2, width 0.12 / 0.3 | 4 / 4 | **0** / **0** | 16 / 18 |
| frame pair_ins 2 / 6 / 12 | 2 each | 100 / 0 / 0 | |
| public frame_example (t=2), merged | 4 / 2 | 100 / 100 | 37 / 8 |
| our Cup 2 frame B | 2 | 100 | 118 |
| sparse 2g/1e, 4g/1e | 4 / 4 | 100 / 100 | 42 / **287** |
| sparse 6g/2e | 4 | 53.8 | 109 |
| sparse 8g+, 12g, 18g, 24g | 4 each | **0** | 9 |
| generic (all sizes, bands, layouts) | 16 | **0** | 10 |
| open_example (default sizes, 72/24) | 8 | **0** | 9 |

Findings:
- f2def is a **Clifford-frame specialist**: it decodes exact quarter-turn frames with ≤ 4 insertions and almost nothing else.
- **Near-Clifford ranging (±0.12 rad) defeats it**, matching the Cup 1 finding that "near a quarter turn is a trap; exactly on it is a gift".
- **Timeout risk:** successful f2def runs can cost 287–514 CPU s.
- Earlier smoke run: our Cup 2 frame A scored 100/100/**0** (lost checkpoint 3). Not yet investigated.

## 6. Defender v1 → v2

### v1: f2def + public-template fitter
Each candidate is a model of the attack circuit; its patch is the exact inverse
(reversed gates, negated angles). All candidates are scored by BIC on all counts
collected so far (Cup 1 rule: extend the pool, never replace). Template stage:
screen all 533 architectures at their angle centres on the 96 most-shot settings,
fit angles of the top 4 by L-BFGS with the adjoint gradient, then refit on all data.

Smoke test (`v1_smoke`):

| Attack | f2def | v1 |
|---|---|---|
| notebook default `mixed` | 0 | **100** |
| notebook default `multilayer` (72 g) | 0 | **97.9** (CP1 ε 1.3e-3, then 100, 100) |
| open_example/23 | 0 | **100** |

### Offline gate pursuit (no template, unknown architecture)

| Attack | gates found | ε | points | CPU s |
|---|---|---|---|---|
| public `zz` (1 g) | 1 | 5.2e-6 | 100 | 4 |
| sparse 4g/1e | 4 | 6.6e-5 | 100 | 7 |
| notebook `open_demo` (5 g) | 5 | 1.3e-4 | 100 | 9 |
| sparse 8g/2e | 10 | 2.4e-4 | 100 | 22 |
| sparse 12g/4e | 17 | 1.6e-4 | 100 | 48 |
| sparse 18g/6e | 17 | 3.2e-4 | 100 | 46 |
| public `mixed` 18g/6e | 21 | 7.1e-4 | 100 | 66 |
| generic 24g/8e | 29 | 4.6e-4 | 100 | 114 |
| public `multilayer` 72g/24e | 54 | 1.0 | **0** | 274 |

Structure learning at 8q is much stronger than expected: greedy pursuit decodes
**unknown** architectures up to at least 24 gates.

### v2: v1 + pursuit source
Pursuit runs from scratch at checkpoint 1 and is warm-continued afterwards, on the
160 most-shot settings, followed by a refit on all data. Budget 120/90/90 CPU s,
with a 600 CPU s encounter cap on optional sources.

## 7. Experiment 2 — attack candidates vs defender v2 (`attack2_v2`, `attack2g_v2`)

| Attack family | n | v2 recovery | CPU s |
|---|---|---|---|
| frame k=2, near-Clifford width 0.3 | 4 | **0** | 493 |
| frame k=24 | 4 | **0** | 472 |
| frame k=48 + 24 pair insertions (no Clifford slot) | 4 | **0** | 384 |
| generic 72g/24e, banded angles, random pairs | 6 | **0** | 520 |
| generic 72g/24e, brickwork | 6 | **0** | 513 |
| generic 72g/24e, all-pairs | 6 | **0** | 501 |
| generic 48g/16e | 4 | **0** | 469 |
| generic 36g/12e | 4 | **0** | 522 |
| generic 24g/8e | 4 | 96.5, 66.7, 0, 0 | 403 |

(The first `attack2_v2` run lost its 30 generic cases to a harness bug: template names
longer than the SDK limit. They were rerun as `attack2g_v2`.)

Findings:
- **Attack phase complete for deep circuits:** every 36/48/72-gate generic candidate and every heavily non-Clifford frame scores 0 against our strongest defender.
- The effective defeat depth for the full defender is **24–36 gates**.
- The 24-gate zeros are **defender bugs**, not learnability limits (offline pursuit scored 100):
  1. **Template overfitting.** Free 72-angle fits of *wrong* public architectures (e.g. `open_example:87:72:24`) beat pursuit's correct model on BIC. Fix: bound template angles to their published ranges.
  2. **Pursuit CPU starvation.** f2def plus templates consume the cap; one encounter fell back to identity.
- v2 CPU is 380–540 s per hard encounter; server limit unpublished (the notebook's local 8q smoke timeout is 900 s).

## 8. Strategic reasoning recorded along the way

- "Won't every team submit a 72-gate circuit?" Gate count is not what decides it; *architecture knowability* is. A copied `multilayer` (72 g) is recovered at 97.9 and a reused Cup 2 frame (72 g) at 100. Private continuous 72-gate circuits tie every defender at 0, so defender work only pays against the minority of recoverable attacks, and it must catch all of them.
- Defender priorities, in order: (a) no crash or timeout; (b) full reliability on defaults, reused frames and short hand-made circuits; (c) push the recoverable depth past 24 gates.

## 9. Bugs found

| Bug | Effect | Status |
|---|---|---|
| Saturated-likelihood constant subtracted instead of added | NLL offset only (no effect on fits or ranking) | fixed |
| Offline scripts imported NumPy before pinning threads | load average 48 on 8 cores | fixed |
| Generic template names > SDK max length | 30 encounters ADMISSION_FAIL | fixed, rerun |
| Wrong-template overfitting wins BIC | 24-gate attacks scored 0 | **open** |
| Pursuit starved by CPU cap | fallback to identity | **open** |
| f2def lost CP3 on Cup 2 frame A | 66.7 instead of 100 | **open** |
| f2def CPU up to 514 s | timeout risk | **open** |

## 10. Plan (updated after the §12 brainstorm)

Ordered by expected value per hour of work. "Rank value" means ranking points against
the likely field; "frontier value" means the largest unknown circuit decoded (write-up).

| # | Item | Mechanism (§12) | Rank value | Frontier value | Cost | Status |
|---|---|---|---|---|---|---|
| P0 | **Make in-defender pursuit match offline pursuit.** ✅ root cause found (§13.1): illegal angles >π silently dropped. v3 still scores 17.3 mean on 24-gate generic (offline: 100). Suspect: f2def's adaptive, shot-concentrated settings. Test offline on f2def's actual panel; if confirmed, switch to a uniform random panel once f2def reports a non-frame | D (measurement) | **high**: short hand-written attacks are the likeliest recoverable opponents | prerequisite for everything below | low | done (v4) |
| P1 | **Seed brute force of public generators** at runtime (`open_example`, `open_multilayer_example`, thousands of seeds, common sizes), screened on a few settings | E | **high**: decodes 72-gate attacks made by generators | none | low | **done**: any open_example seed decoded (e.g. 537 → 97–99) |
| P2 | **Subspace/beam pursuit with rearrangement moves** (delete, move, swap, re-axis), several candidates per step, prune back | A | medium | **high**: target 36g/12e, then 48g/16e | medium | done: p1 + p2 in pool; adaptive back-loading + random panel → 36g often, one 48g decoded at CP3 |
| P3 | **Dictionary pruning by measured coupling graph** (restrict pair gates to pairs with evidence of coupling) | C1 | medium | medium: shrinks each step's search ~3× | low | deprioritized (§13.3) |
| P4 | **Residual pursuit:** at checkpoints 2–3 measure with the current best inverse as the analysis circuit, run pursuit on the shallower residual | D | low–medium | potentially high | medium–high (uses settings) | experiment only |
| P5 | **Relax-then-project:** overparameterized layered ansatz + L1, prune to legal gates | B | low | unknown | high | only if time |
| P6 | Clifford-doped extension of f2def past 4 insertions | F | low | low | high | deferred |
| P7 | Multithreaded kernels | G | depends on runner | medium | low | blocked: runner limits unknown |
| R1 | **Timeout safety:** f2def alone up to 514 CPU s; v3 generic cases reach ~690 s | Inversion | **critical** | — | low | mitigated: f2def fit time-boxed, 600 s encounter cap; broad max 597 s (limit unpublished) |
| R2 | Final attacks: verify vs v3, stock, and an overpowered adversary; build and validate the OPEN8 ZIP | — | **critical** | — | low | **done**: v10 VALIDATED_LOCALLY, attacks 0 vs every defender |

Rule for every item: develop offline first, then add as a *new pool source* (never
replace), then pass the regression set (`specs_reg3.json`) without losses before it
ships.

---

## 11. Championship strategy

### 11.1 How the score works, and what that implies

Each pairing scores `(our recovery + (100 − their recovery)) / 2`. That gives two
independent halves:

- **Attack half:** how badly opponents fail to undo our circuit. Maximum 50, when every opponent recovers 0.
- **Defence half:** how well we undo theirs.

Two facts from our data decide where the effort should go:

1. **A private, deep, continuous-angle attack scores 0 against every defender we could
   build**, including ones given 8× our CPU budget (being verified, §11.2). The
   organizers say so too: unrecoverable templates are legal, and "ranks separate on the
   attacks that can be recovered". So the attack half is easy to max out, and
   serious teams will max it out too. That makes it a **tie-level requirement, not a
   differentiator**.
2. **The defence half is where ranks separate.** Against a team with a private deep
   attack, nobody scores and everyone ties. Against a team whose attack can be
   recovered, only defenders that recover it gain points. The winning defender
   therefore catches **every recoverable attack in the field**, reliably and within the time
   limit, and doesn't waste CPU on attacks nobody can solve.

The Cup 2 feedback shows who we're playing: 11 of 12 teams recovered 0 against our
frames (weak defenders, and likely also simpler or copied attacks), and one team
(`9d05da`) recovered everything. Our rank against `9d05da` is decided almost
entirely by who recovers more of the *other eleven* teams' attacks.

### 11.2 Attack strategy

**What makes an attack unrecoverable at 8 qubits.** A defender has to learn the attack's
*architecture* (which gate, which qubits, in what order) as well as its angles. Cup 1
showed that once the architecture is known, the angles are always learnable (100
points). So the attack's job is to make the architecture unlearnable:

| Property | Why it matters | Measured evidence |
|---|---|---|
| **Private architecture** | Anything copied from a public helper is identified by our template fitter, and a rival can build the same thing | notebook `mixed` and `multilayer`: 0 → 100 / 97.9 once templates were added |
| **Depth: at or near 72 gates / 24 entanglers** | Greedy structure search decodes unknown circuits up to ~24 gates; beyond that it commits to a wrong structure | pursuit: 100 at ≤ 24 gates, 0 at 36, 48 and 72 |
| **No Clifford skeleton** | Clifford-frame learners (f2def, likely `9d05da`) exploit exact quarter turns and decode frames with ≤ 4 non-Clifford insertions | f2def: 100 on frames with k ≤ 4, 0 at k ≥ 6 |
| **Angles away from 0, ±π/2 and π** | Angles near 0 are nearly identity (a defender can drop them cheaply); near ±π/2 or π they're (near-)Clifford and algebraically exploitable | bands [0.6,1.2] ∪ [1.9,2.5] rad |
| **Entanglers spread across many pairs** | Prevents the circuit splitting into small independent blocks that a local learner could solve separately | brickwork uses 8 pairs, all-pairs uses 17 |

**The two submitted templates** (`quantum_duel_work/quantum-duel-8q-open-0.7.1/attacks.json`):

| Slot | Name | Layout | Size |
|---|---|---|---|
| 1 | Deep continuous brick | entanglers in an alternating nearest-neighbour brickwork | 72 gates / 24 entanglers |
| 2 | Deep continuous all-pairs | entanglers on random pairs from all 28 | 72 gates / 24 entanglers |

- Both are generated from private seeds.
- The rules require two templates that are *structurally distinct*; different layouts guarantee that and hedge against any learner that happens to suit one layout.
- Both pass the server qualifier with **64/64** readiness draws.
- These families scored **0 in all 18 encounters** against our best defender (v2).
- The exact final seeds are being verified against v3, stock `recovery8`, and an overpowered adversary (96k shots at once, 720 settings, 1,200 CPU s pursuit).

**Rejected alternatives, and why:**
- *Reusing Cup 2 frames.* `9d05da` recovered them at 100.
- *Near-Clifford frames* (angles ±0.3 around quarter turns). They also scored 0 against us, but they keep a Clifford skeleton, which is exactly what a strong frame learner is built around. That's an unnecessary risk when generic circuits score the same 0.
- *Short or clever attacks.* Pursuit solved every attack up to 24 gates, including hand-built ones.

### 11.3 Defence strategy

**Design rule (carried over from Cup 1): extend the pool, never replace it.** The
defender runs several independent *sources*. Each proposes a model of the attack
circuit, and its patch is that model's exact inverse. All candidates are scored on
**all counts collected so far**, and the best (by BIC) is submitted at each checkpoint.
A new source can only change the outcome by explaining the data better, so adding
sources cannot make us worse except through selection noise.

| Source | Catches | How it works |
|---|---|---|
| **f2def** (Cup 2 learner, unchanged) | Clifford frames with few non-Clifford insertions, including reused Cup 2 attacks and public frame examples | parity learning of the Clifford frame + residual fit; 100/100 in the official Cup 2 round |
| **Public-template fitter** | anything built from public helpers or the notebook: default `mixed`/`multilayer`, `open_example` (seeds 0–127, three sizes), frame examples, `open_demo` | screens 533 known architectures at their angle centres, then fits the angles by count likelihood (adjoint gradient), **bounded to the published ranges** |
| **Greedy gate pursuit** | hand-written or unknown circuits up to ~24 gates | one gradient sweep scores all 108 candidate rotations at every insertion point; line-search the best, insert it, refit all angles; stop on BIC. Runs from scratch at each checkpoint plus a warm continuation |
| identity / previous patch | safety floor | never go backwards |

**What pursuit is (plain-language explanation).** Pursuit is **forward stepwise
regression where the features are quantum gates**. The name comes from *matching
pursuit* in signal processing, which uses the same greedy idea.

| Stepwise regression | Pursuit |
|---|---|
| candidate features (columns) | 108 possible gates: `rx/ry/rz` on each of 8 qubits (24) + `rxx/ryy/rzz` on each of 28 pairs (84) |
| coefficients | rotation angles |
| fit criterion (squared error) | likelihood of the observed counts under the circuit (computed by `sim8.py`) |
| add the feature that most reduces error | add the gate that most increases likelihood |
| stop when the next feature is not significant (AIC/BIC) | stop when the next gate gains less than the BIC penalty (~10 log-likelihood units at 32k shots) |

One difference makes it harder than regression: **order matters**, because gates don't
commute. Every candidate gate must be tried at every position in the current circuit,
about 108 × 21 ≈ 2,300 options at 20 gates. One step:

1. **Score every option at once.** An adjoint pass (the same idea as backpropagation)
   gives the likelihood slope for inserting each gate at each position with a tiny
   angle. This is like computing every feature's correlation with the residual.
2. **Try the top 10 properly.** Line-search each one's angle and measure the real gain.
3. **Insert the winner and refit all angles jointly** (L-BFGS), as stepwise regression
   re-estimates all coefficients after each addition.
4. Repeat until the gain drops below the BIC penalty.

The final circuit is the attack model; the patch is that circuit reversed with negated angles.

**Why it stops working around 24–36 gates.** Like greedy stepwise regression with
correlated features, pursuit **cannot undo an early mistake**. In shallow circuits the
strongest single signal really is a true gate. In deep circuits the true gates' effects
are so entangled that the best-looking early gate is often wrong, and pursuit then fits
a confident wrong structure (54 wrong gates on the 72-gate `multilayer`). v3 therefore
restarts pursuit from scratch at each checkpoint with 2× and then 3× the data, and keeps
the warm continuation as a separate candidate.

**Measurement.** f2def chooses the settings (its parity learning needs adaptive
choices), and any leftover budget is spread over those same settings. Cup 1 showed
that clever shot allocation is worth at most ~2 points; what matters is that every
source **uses all the data**. v2's pursuit dropped to the 160 most-shot settings, which
turned 100 into 0 on 24-gate attacks (§ablation below).

**Controlling CPU.** The server's time limit is unpublished; the notebook's local 8q
smoke timeout is 900 s.
- **Goodness-of-fit gate:** before running expensive sources, we compute a z-score of the observed log-likelihood against its expectation under the best model. Calibrated offline: true model z ≈ N(−0.05, 0.8); patch-quality fit ≤ 3; wrong models 40–200. If z < 6, the data is already explained and pursuit is skipped. That saves its budget on frames and copied templates.
- **Budgets:** templates ≤ 60 s per checkpoint (full screen only at checkpoint 1); pursuit ≤ 150 s per checkpoint; optional sources stop at a 700 s encounter cap.
- **Robustness:** every source is wrapped in try/except, a legal patch is submitted at every checkpoint, and each patch is validated before submission.

**What the defender deliberately does *not* try to do.** Recover private generic
circuits of 36+ gates. Nothing we built does, the overpowered adversary test checks
whether more compute changes that, and those encounters tie every team anyway.

### 11.4 Supporting ablation: pursuit needs all the data

Same two failing 24-gate attacks, offline:

| Pursuit's data | seed 22 | seed 21 |
|---|---|---|
| random 360 settings, all 32k shots | 100 | 100 |
| random 720 settings, all shots | 100 | 100 |
| 720 settings, only the 160 most-shot (v2 behaviour) | 67 | **0** |
| random 160 settings, all shots | 100 | 0 |

### 11.5 Defender versions

| Version | Change | Result |
|---|---|---|
| f2def | Cup 2 learner as-is | 0 on every continuous-angle attack, including both notebook defaults |
| v1 | + public-template fitter | defaults: `mixed` 100, `multilayer` 97.9, open_example/23 100 |
| v2 | + pursuit (on a 160-setting subset) | 24-gate generic: 96.5 / 66.7 / 0 / 0 (two selection and data bugs) |
| v3 | pursuit on all data, fresh + warm; bounded templates; gof gate; budgets | regression run in progress |

### 11.6 Open risks

1. **Server time limit unknown.** f2def alone reached 514 CPU s on k=4 frames. If the limit is tighter than ~600 s, f2def's synthesis budgets need cutting.
2. **Rivals with a stronger unknown-architecture learner** could recover attacks between 24 and 36 gates that we can't. That only affects our defence against such attacks, and our own attacks are 72 gates.
3. **f2def lost checkpoint 3 once** on our Cup 2 frame A (100/100/0). Not yet investigated.

---

## 12. Frontier: how large an unknown 8q circuit can be decoded? (brainstorm + prior art)

**Short answer.** Nothing shows that decoding 36+ gates is impossible. The failure is in
our search, not in the data:
- A 72-gate circuit with a *known* architecture was recovered at 97.9 from the same shots.
- Theory says the number of samples needed grows only linearly with gate count.

Prior art on random all-to-all circuits nevertheless puts a learnability transition
around 4–5 entangling layers at n = 8, *even with the layout given*. So a realistic target
is ~16–18 entanglers (~48 gates), and a private generic 72g/24e circuit is likely out of
reach for anyone within the time limit.

### Prior art
| Work | Result | What it means here |
|---|---|---|
| Zhao, Lewis, Kannan, Quek, Huang, Caro, *Learning quantum states and unitaries of bounded gate complexity* (PRX Quantum 2024, arXiv:2310.19882) | Samples needed scale **linearly** in gate count G; computation must scale as exp(Ω(min(G, n))) under crypto assumptions | Samples: 96k shots is not the bottleneck for 72 gates. Compute: the lower bound is exp(min(G, n)) = exp(8) here, so asymptotic hardness does not bind at 8 qubits. The binding limit is our minutes of CPU |
| Huang, Liu et al., *Learning shallow quantum circuits* (arXiv:2401.10095) | Polynomial-time learning of unknown-architecture shallow circuits from single-qubit measurements, via **local inversions** + **circuit sewing** | Sewing needs n extra ancilla qubits, and our patch has none. The local-inversion search itself needs queries of UV (V before U), but our oracle only allows analysis circuits *after* U. Only partially transferable |
| *Proper learning of shallow all-to-all quantum circuits* (arXiv:2608.20162) | Iterative local gate inversions from the **front and back**; sharp learnability transition at depth d* ≈ log₂n + log₂log₂n, from light-cone growth. **Assumes the gate layout is known** | n = 8 gives d* ≈ 4.6 layers of 4 pair gates ≈ 18 entanglers. Our attacks (24 ent) are past it; our pursuit fails at 12 ent (36 g), below it, so there is headroom |
| Grewal, Iyer, Kretschmer, Liang; Leone, Oliviero, Hamma: learning t-doped stabilizer states (arXiv:2305.13409, 2308.07014) | Clifford + t non-Clifford gates learnable in poly(n, 2^t) | Theory behind f2def's regime; extends frame learning to more insertions (~6–10 at n = 8) |
| Larocca et al., *Theory of overparametrization in QNNs* (Nat. Comput. Sci. 2023) | Spurious local minima disappear once parameters exceed the dynamical Lie algebra dimension | For generic 8q gates that dimension is 4⁸ − 1 = 65,535, so full overparametrization is unreachable. Partial relaxation may still smooth the landscape (untested) |

### Mechanisms (after collapsing duplicate ideas)
| # | Mechanism | Concrete forms | Status |
|---|---|---|---|
| A | **Better combinatorial search** on the same likelihood | beam search; delete/move/swap/re-axis moves (phylogenetics SPR/NNI); subspace pursuit (add several, refit, prune); stochastic restarts; MCMC over structures | strongest near-term candidate |
| B | **Relax, then project** | overparameterized layered ansatz with L1 / group-lasso, then prune to legal gates; Cup 1's dense→project two-stage | plausible; untested at 8q |
| C1 | **Shrink the dictionary with structure read from the data** | entangler-pair topology from coupling statistics (Cup 1 §5.5: 3–4/4 pairs identified) | cheap; multiplies A |
| C2 | **Decompose into per-qubit subproblems** | local inversions (Huang et al.) | blocked: no ancillas, analysis circuits only after U |
| D | **Change what is measured: residual pursuit** | at checkpoints 2–3, apply the current best inverse as the analysis circuit; the residual V·U is shallower if V is partly right, so pursuit works on an easier problem (boosting on residuals) | novel, risky; Cup 1 found closed-loop hurts *precision*, but here the aim is *structure* |
| E | **Prior over opponents' generators** | brute-force `open_example` / `open_multilayer_example` over thousands of seeds and common sizes at runtime | decodes 72-gate attacks, but only generator-made ones |
| F | **Clifford-doped algebra** | extend f2def past 4 insertions | separate regime (near-Clifford attacks) |
| G | **More compute** | multithreaded Numba if the runner allows | depends on unknown runner limits |

Rejected: full process tomography (65,535 parameters; ~0.7 infidelity at 96k shots, zero points), learned neural decoders (no training data or time), state across repeated encounters (not available to a rated run).

### Recommendation
- **Build A + C1:** subspace/beam pursuit with rearrangement moves, dictionary pruned by the measured coupling graph, restarted per checkpoint, gated by the goodness-of-fit test and CPU budgets.
- **Target:** 36g/12e reliably, then 48g/16e. Measure with the existing offline harness before touching the defender.
- **Strongest case against:** opponents' attacks may be bimodal, either defaults and short circuits (already solved) or 72-gate scramblers (unsolvable). Then pushing 24 → 48 gates earns zero ranking points and adds timeout risk, and E (seed brute force) or reliability work may earn more per hour.

Sources: [arXiv:2310.19882](https://arxiv.org/abs/2310.19882), [arXiv:2401.10095](https://arxiv.org/abs/2401.10095), [PennyLane local-inversion demo](https://pennylane.ai/qml/demos/tutorial_learningshallow), [arXiv:2608.20162](https://arxiv.org/abs/2608.20162), [arXiv:2305.13409](https://arxiv.org/pdf/2305.13409), [arXiv:2308.07014](https://arxiv.org/pdf/2308.07014), [Larocca et al. 2023](https://www.nature.com/articles/s43588-023-00467-6).


---

## 13. Iteration log, Thursday evening (v3 → v4)

### 13.1 P0 root cause: pursuit's correct models were being thrown away
- **Symptom:** v3 scored 17.3 mean on 24-gate generic, while offline pursuit scored 100.
- **Ruled out, measurement panel:** pursuit run offline on f2def's actual checkpoint-1 panel (~494 settings × 65 shots, nearly uniform) scored 100, 100, 54.5, 35. The last two only ran out of time.
- **Instrumented encounter** (`dev/open8/diag_enc.py`): at checkpoint 1 pursuit found a *correct* 29-gate model (NLL 22,511, the same as offline successes), yet it never appeared in the pool.
- **Cause:** pursuit refits angles without bounds, so some drift past ±π. The rules reject patch angles outside [−π, π] (`validate` → "Rotation angle outside the round's canonical range"), and the pool's legality filter silently discarded the model. Offline scoring never validated angles, which is why offline always looked fine.
- **Fix:** wrap every model angle into [−π, π]. R(θ + 2π) = −R(θ) is a global phase, so the fix is physically exact.

### 13.2 Frontier benchmark (offline, 32k shots, 360 random settings, 300 CPU s)
| Generic attack | p1 (gradient-ranked line search) | p2 (closed-form insertion + deletion) |
|---|---|---|
| 24g / 8e | 95.5, 84.0, 100 | 100, 93.7, 69.1 (about 2× faster: 43–105 s) |
| 36g / 12e | **100, 100**, 0 | 62.0, **100**, 20.3 |
| 48g / 16e | 41.2, 0, 0 | 0, **99.1**, 12.1 |

- Pursuit2's closed-form insertion curve (inserting R_P(θ) at p gives amplitudes cos(θ/2)A − i sin(θ/2)B) was verified to match explicit simulation to 4 decimals. It makes each candidate ~15× cheaper, so 40 candidates are scored exactly per step instead of 10.
- **The frontier is past 36 gates at checkpoint-1 data, and 48 gates is sometimes reachable.** The two engines win different cases, so v4 runs both (p2 fresh at checkpoints 1 and 3, p1 fresh at checkpoint 2, p2 warm continuation), and the pool picks.

### 13.3 P3 deprioritized
With p2 the per-step cost is no longer dominated by candidate scoring. For deep circuits every qubit's light cone covers all 8 qubits, so a coupling statistic stops identifying *direct* gate pairs. Little speed and little accuracy to gain.

### 13.4 Final attacks verified (the exact seeds in `attacks.json`)
| Opponent | Encounters | Recovery |
|---|---|---|
| our defender v3 (f2def + templates + pursuit) | 6 | **0** (all checkpoints) |
| stock `recovery8` | 4 | **0** |
| overpowered adversary: all 96k shots at once, 720 settings, top-20 pursuit, 1,200 CPU s | 4 | **0** (found 61–64 wrong gates; ε ≈ 1.0) |

### 13.5 v4 changes
1. Angle wrapping (13.1).
2. **Generator source (P1):** replays the SDK's `open_example` (sizes 36/12, 72/24, 18/6; seeds 128–999, then 23 "special" seeds × 8 sizes, then 0–299 × 5 extra sizes) and `open_multilayer_example`. Screened on the 8 most-shot settings: in a 75-seed test the true seed ranked **first** with ~2× NLL separation. ~12 ms per candidate, checkpoint 1 only, 60 s budget, only when nothing explains the data.
3. **Template/generator admission by goodness of fit:** a template enters the pool only if its fit has z < 10. This removes the "wrong 72-gate template wins BIC" failure. A bounded fit is tried first, with an unbounded retry so a correct architecture with *edited* ranges is still accepted.
4. **Early exit:** if f2def or the previous patch already explains the data (z < 6), all optional sources are skipped at that checkpoint.
5. Budgets: templates 45 s; pursuit 150/130/110 s; optional sources stop at 480 CPU s per encounter (v3 reached 715 s).

### 13.6 v3 regression (for reference; superseded by v4)
| Family | v3 mean | CPU s |
|---|---|---|
| Cup 2 frame A (previously lost checkpoint 3) | 100 ×3 | 378 |
| frame k=2 / k=4 / public frame | 100 / 100 / 99.7 | 247 / 652 / 596 |
| `mixed` / `multilayer` / open_example 23 | 100 / 98.7 / 99.5 | 166 / 230 / 214 |
| `open_demo` | 100 | 827 |
| generic 24g / 36g | 17.3 / 0 | 524 / 692 |
| sparse 12g / 18g | 7.5 / 23.6 | 470 / 515 |

### 13.7 Submission candidate
`dev/build.py --profile quantum-duel-8q-open-0.7.1 --files main.py opendef.py sim8.py pursuit.py f2def.py tlib.json --no-smoke` gives **STATIC_VALIDATED, valid_for_upload: True**, 222 KB zipped / 821 KB expanded (limits 2 MB / 4 MB; `tlib.json` 725 KB < 1 MB per file after rounding angles to 1e-4). Numba kernels use `cache=False` so a read-only runner directory cannot break them.

### 13.8 v4 regression (`reg4_v4`, 57 encounters, 0 crashes, mean 75.0)
| Family | v3 | v4 | CPU s |
|---|---|---|---|
| generic 24g/8e | 17.3 | **90.6** | 248 |
| sparse 18g/6e | 23.6 | **100** | 124 |
| sparse 12g/4e | 7.5 | 59.3 | 140 |
| **open_example seed 537, 72g/24e** (in no list; generator source) | — | **97.1** | 102 |
| open_example 23 and seed 2026 | 99.5 | 99.2 | 84 |
| `mixed` with edited ranges (±0.3 wider) | — | 59.2 | 226 |
| generic 36g / 48g | 0 / — | 16.7 / 0 | 544 / 565 |
| frames k=2 / k=4 / public frame / Cup 2 frame A | 100 | 100 | 40 / 478 / 400 / 135 |
| `mixed` / `multilayer` | 100 / 98.7 | 100 / 96.0 | 81 / 89 |
| `open_demo` (5 g) | 100 | 100 | **853** |

**P1 confirmed:** the generator source decoded a 72-gate attack built from a seed that appears in no list.

### 13.9 Diagnoses from v4 logs → v5
1. **"Explained" too loose, and never refit.** At 32k shots the goodness-of-fit test cannot tell ε ≈ 0.01–0.02 from ε ≈ 0 (expected z ≈ N·ε·c / √(2·nnz) ≈ 3–7). A generator fit started from wrong angle centres passed z < 6 at checkpoint 1 with ε ≈ 0.02. Every later checkpoint then skipped all sources, and the angles were never refitted on the extra 64k shots (flat 36.3 / 36.3 / 36.3). **Fix:** skip threshold z < 3, and at every checkpoint refit the two best models' angles on all data (60 L-BFGS iterations).
2. **f2def's own CPU.** Profiling `open_demo`: 238 of 252 CPU s at checkpoint 1 is `Algebra.fit` with 40 fresh starts (our Cup 2 hardening), repeated every checkpoint. **Fix:** warm start first, then fresh starts in chunks of 4 (distinct seeds) until a 45 CPU s fit deadline.
3. **36-gate generic in-defender (16.7) vs offline (100, 100, 0 at 300 s):** budget-limited. This motivates the back-loading measurement (§14).

---

## 14. Back-loading: sacrificing early checkpoints for the last one

**Question:** is there a strategy that gives up 100 on every checkpoint but secures points on the last?

**What can and cannot move.**
- **Shots cannot move.** Blocks are use-or-lose (`available_now = (stage+1)·block − spent − forfeited`), and checkpoint 3 sees the same 96k shots whatever patches were submitted earlier. Patches never affect the data.
- **CPU (per-encounter limit) and measurement design can move.**
- **Nothing guarantees points on a deep private attack:** our 72-gate attacks scored 0 even against an adversary with 96k shots and 1,200 s.

**Points arithmetic.** Solving at checkpoint 1 = 100, at checkpoint 2 = 66.7, only at checkpoint 3 = 33.3. Back-loading can gain at most +33.3 per encounter, and only on attacks that fail with a split budget but succeed with a concentrated one. It loses on everything front-loading solves early.

**Measurement** (offline, generic banded attacks, `dev/open8/data/backload.log` vs `frontier.log`):

| Generic attack | p1 @ 32k / 300 s | p1 @ 96k / 400 s | p2 @ 32k / 300 s | p2 @ 96k / 400 s |
|---|---|---|---|---|
| 24g / 8e | 95.5, 84, 100 | — | 100, 93.7, 69.1 | — |
| 36g / 12e | 100, 100, 0 | **100, 100, 100** (260–358 s) | 62, 100, 20.3 | 31.8, 35.9, 3.6 |
| 48g / 16e | 41.2, 0, 0 | 3.7, 0, 33.2 | 0, 99.1, 12.1 | 0, 36.1, 0 |
| 60g / 20e | — | 0, 0, 0 | — | 0, 0, 0 |

**Conclusions.**
- The concentrated budget makes **36-gate attacks reliably solvable** at checkpoint 3 (3/3 with p1).
- 48 gates stays out of reach even with the full budget, and 60 is hopeless. The practical frontier is ~36–48 gates.
- p2's closed-form speed does not carry over to 720 settings; p1 is the right engine for checkpoint 3.
- **Pure back-loading is wrong:** 24-gate attacks are solved at checkpoint 1 in 45–100 s, and waiting would give up two-thirds of their points.

**Adopted: adaptive back-loading (v6).**
- Checkpoint 1 runs the cheap sources plus a p2 pursuit (110 s).
- If the stage-1 winner does not explain the data (z > 3), the encounter enters **hard mode**: checkpoint 2 gets only a 30 s warm continuation, and checkpoint 3 gets a fresh p1 pursuit on all 96k shots with the rest of the encounter budget (cap raised 480 → 600 CPU s, 40 s reserved for the final refit).
- Risk is low: in hard mode checkpoints 1–2 have usually scored ≈ 0, so spending late puts little at stake.

### 14.1 v5 regression (`reg5_v5`, 26 encounters, 0 crashes, mean 98.3, max CPU 279 s)
| Family | v4 | v5 | CPU s v4 → v5 |
|---|---|---|---|
| `open_demo` | 100 | 100 | 853 → **272** |
| `mixed`, edited ranges | 59.2 | **100** | 226 → 78 |
| sparse 12g/4e | 59.3 | **94.4** | 140 → 139 |
| generic 24g/8e | 90.6 | 95.1 | 248 → 141 |
| frame k=2 / k=4 / public frame | 100 / 100 / 100 | 98.1 / 99.0 / 99.1 | k=4: 478 → **261** |
| Cup 2 frame A / merged / `mixed` / `multilayer` / open_example 537 | ≥ 96 | 99.9 / 99.7 / 100 / 98.0 / 99.0 | 23–222 |

The refit fixed the flat partial scores, and the time-boxed f2def fit removed the ~850 s worst case. Cost: slightly lower checkpoint-1 scores on some frames (e.g. 88.7 → 100 → 100) from fewer fresh starts.

### 14.2 v6 hard-set result and the real CPU leak
| Generic | v4 | v6 (adaptive back-loading) | CPU s |
|---|---|---|---|
| 24g | 90.6 | 95.1 | 192 |
| 36g | 16.7 | 26.2 | 471 |
| 48g | 0 | 0 | 564 |

- Hard mode triggered correctly, but checkpoint 3's p1 pursuit got only **64–110 s** (intended ~300), because checkpoints 1–2 had already used ~450 s.
- **Cause:** screening is time-boxed but the fits after it were not. The generator fitted its top 3 seeds and the template source its top 4, each a 72-parameter fit of up to 200 iterations on all data, and templates repeated this every checkpoint. On a non-template attack every one of those fits is wasted (z ≈ 160).
- **Screening separation** (best NLL ÷ next *different* architecture's), measured from logged screens:

| Case | Ratio |
|---|---|
| true template or generator match (`mixed`, `multilayer`, open_example, edited `mixed`, sparse 18/6 via generator) | **0.23–0.56** |
| no match (generic 24/36/48 unknown architecture) | **0.95–0.99** |

**v7 fix:** fit a template or generator candidate only if separation < 0.8, fit only the best, and never refit a rejected candidate. Per-checkpoint CPU is now logged.

---

## 15. New attack designs probed, and the few-qubit decoder

### 15.1 Probes (v6, `probe7_v6`)
| Deep 72g/24e attack | v6 recovery (4 encounters) |
|---|---|
| small angles 0.1–0.4 rad, all 8 qubits | 0, 7.2, 35.3, 27.0 (pursuit finds approximate compact models) |
| confined to **3 qubits** | 9.0, 20.1, 0, 4.5 |
| confined to 4 qubits | 0 ×4 |

### 15.2 Few-qubit decoder (`subsys.py`, v8)
- **Idea.** An attack confined to k ≤ 3 qubits is a k-qubit unitary (63 parameters at k = 3) *whatever its depth*. A fixed circuit with more parameters than that fits inside the patch caps (3 qubits: 16 blocks = 89 gates, 16 entanglers). By overparametrization theory (Larocca et al. 2023) its landscape has no spurious minima, so no structure search is needed.
- **Expressivity check** (noise-free fit to exact 72-gate 3-qubit targets, `dev/open8/t_ansatz3.py`): 14 blocks (79 gates) and 19 blocks (104 gates) both reach **ε ≈ 1e-8 from every random start** (12/12).
- **Support detection is exact.** The oracle has no noise beyond shot noise, so an untouched qubit prepared and read in the same basis never disagrees; ≥ 2 disagreements mark a qubit as touched. Correct on 4/4 test attacks (3-, 4- and 8-qubit supports).
- **Fit on the support's marginal counts** (k-qubit data, 32× cheaper than 8-qubit simulation), 4 random restarts.
- **Offline, f2def's checkpoint-1 records (32k shots):** deep 3-qubit attacks **85.2 and 86.9 points in 2–3 CPU s** (v6: 0–20). The residual ε ≈ 2e-3 is shot noise on 89 parameters, so it should reach 100 at 64k–96k shots.
- **Also:** pursuit's dictionary is now restricted to the detected support whenever the attack leaves qubits idle. This is a sound form of P3: exact, unlike the coupling statistic.
- **4-qubit support stays open:** SU(16) has 255 parameters, beyond the 108-gate patch cap, so this trick does not extend.

### 15.3 v7 results, and why the absolute goodness-of-fit test was the wrong tool
**v7** (`v7`, 26 encounters): the CPU leak is gone. Checkpoint 1 takes ~125–134 s, checkpoint 2 ~170 s cumulative, and hard mode's checkpoint-3 pursuit gets **~380 s** as intended. Copied templates are now cheap (`mixed` 81 → 15 s, `multilayer` 93 → 26 s, open_example/537 97 → 40 s). Remaining problems:

| Case | v7 | Issue |
|---|---|---|
| 36g hard mode (4 enc.) | 0, 0, 18.3, 78.7 | p1 with 380 s builds 39 wrong gates, although 3/3 offline at the same budget → measurement-panel test (§15.4) |
| 36g seeds 22/23 (not hard) | 83.5 → 87.6; 57.9 → 58.1 | model "explained" at z < 3 but ε ≈ 0.003–0.005; later checkpoints skipped search with ~470 s CPU unused |
| edited-range `mixed` | 60.7 (v5: 100) | see below |

**Diagnosis** (f2def's checkpoint-1 panel, offline):

| Edited `mixed` fit | NLL | z | ε | Points |
|---|---|---|---|---|
| bounded to published ranges | 21,874 | **1.7** | 7.8e-2 | **5.3** |
| free refit | 20,022 | −0.2 | 2.7e-4 | 100 |
| fit started from the true angles | 20,022 | −0.2 | 2.7e-4 | 100 |

Nine of the 18 true angles lie outside the published ranges. A 1,852-unit likelihood gap, which is overwhelming evidence, moves z only from −0.2 to 1.7, because z divides by the noise of ~15,000 count bins and dilutes a concentrated misfit. **Comparing likelihoods between models is sharp; the absolute z test is not.** It remains useful only to *reject* wildly wrong models (z ≫ 10).

**v9 changes.**
1. Template fits always add the free refit; the pool's likelihood comparison decides.
2. "Explained, skip further search" applies only to exact-structure winners (f2def lossless compile, template, generator, subsystem), where skipping saves real CPU on frames.
3. A pursuit-built winner always gets a 60 s warm continuation at later checkpoints. The warm pursuit stops on BIC within seconds when the model is already right.

### 15.4 v8 results (`v8`, 30 encounters)
| Attack | v6 | v8 | CPU s |
|---|---|---|---|
| **72-gate deep, confined to 3 qubits** (6 enc., incl. support {2,5,7}) | 0–20 | **94.3–97.0** (CP1 84–91, then 100) | 40–50 |
| 72-gate deep, 4 qubits | 0 | 0 | ~569 |
| 36-gate on 5 qubits | — | 3.0, 8.7 | ~520 |
| 72-gate small angles | 0–35 | 5.1–34.5 | ~560 |
| generic 36g | 26–30 | 67.6 [3,100,100]; 33.3 [0,0,**100**] | 172, 447 |
| generic 24g, seed 21 | 88–98 | 81.6, 58.4 | 140, 224 |
| `mixed` / `multilayer` / `open_demo` / frame k=2 / sparse 12 | ~94–100 | 100 / 98.0 / 100 / 100 / 94.4 | 14–232 |

**Frontier statement:** an attack confined to ≤ 3 qubits is now decoded at **any depth**.

### 15.5 Measurement panel diagnosis → v10
- **Offline test** (`dev/open8/panel3.py`): 36g, p1 with 380 s at 96k shots.

| Panel | Seed 21 | Seed 22 |
|---|---|---|
| f2def's adaptive panel, all three checkpoints (712 settings) | 0 | 18.3 |
| f2def's checkpoint 1 (492 settings), then 228 random settings | 0 | 16.9 |
| **uniform random, 720 settings** | **100** | **100** |

- **Why:** f2def's parity learning measures in *families*, one random base setting plus 8 single-sign-flip variants (`0 +i 0 0 …` / `1 +i 0 0 …` / `0 -i 0 0 …`, same readout basis). That is ideal for learning Clifford parities and poor for likelihood search, which needs diverse directions. On non-frames the loop never completes and uses 492 of the 720-setting budget at checkpoint 1.
- **How many settings frames need** (`dev/open8/f2cap.py`):

| Frame | 90 settings | 180 | 270 |
|---|---|---|---|
| k=2 | ✗ (rank 9) | **100** (rank 13) | 100 |
| Cup 2 frame A | ✗ (8) | **100** (13) | 100 |
| merged | ✗ (9) | **100** (16) | 100 |
| k=4 | ✗ (8) | ✗ (11) | **100** (12) |
| public FRAME4 | ✗ (5) | ✗ (10) | **100** (12) |

- **Classifier:** raw parity rank misleads because each idle qubit adds 2 trivial parities (`open_demo` rank 10, 3-qubit attack rank 10 at 90 settings). **Rank on the support** = rank − 2 × idle qubits: frames 10–16 at 180 settings, non-frames ≤ 4.
- **v10 measurement:**
  - Checkpoint 1: f2def probe with 180 settings / 16k shots.
  - If it completed, or the support rank is ≥ 8, the attack is frame-like: f2def gets the rest of the block, as before.
  - Otherwise the rest of the block goes to ~180 new uniformly random settings, and f2def is skipped at checkpoints 2–3, which get ~180 new random settings each (≈540 diverse settings in total).

### 15.6 v10 results (`v10`, 34 encounters, mean 84.6, 1 zero, 0 crashes, max CPU 565 s)
| Group | v10 | CPU s |
|---|---|---|
| Frame probe classifier | **correct on all 17 attack types** (frames incl. k=4 / FRAME4 via support rank 10–11 → f2def; everything else → random panel) | — |
| Frames: k=2, k=4, FRAME4, Cup 2 frame A, merged | **100 on 10/10** | 23–234 |
| `mixed` / `multilayer` / open_example 537 / edited `mixed` / `open_demo` / open_example 31 | 100 / 97.4–99.3 / 97.7–98.6 / **100** / 100 / 100 | **7–29** |
| generic 24g / sparse 12g | **100, 100 / 100, 100** | 47–154 |
| generic 36g (6 enc.) | 33.3 [0,0,**100**] ×3, 6.8, 71.6, 75.9 | 138–564 |
| **generic 48g** | 0, **33.3 [0,0,100]** | 531–565 |
| 72g deep on 3 qubits | 99.2, 97.5 | 34–39 |

The random panel lets hard mode's checkpoint-3 pursuit decode 36-gate attacks, and **one 48-gate attack**.

### 15.7 Validated submission (v10)
`dev/build.py --profile quantum-duel-8q-open-0.7.1 --version v10 --files main.py opendef.py sim8.py pursuit.py subsys.py f2def.py tlib.json --cases local zz mixed multilayer --timeout 900`

- **VALIDATED_LOCALLY, valid_for_upload: True**, 226,810 bytes zipped / 834,437 expanded.
- The packaged code ran in fresh processes on the notebook's smoke cases: local 100, zz 100, mixed 100, multilayer 97.37.
- Attack qualifier: VALIDATED.
- Frozen copy: `quantum_duel_work/quantum-duel-8q-open-0.7.1/submission_v10_validated.zip`, sha256 `0a1cda0d1af1ec28c5356f94f5267f7b764047e2797e1a634ca2e526aaccbeec`.


### 15.8 Broad validation of v10 (`broad_v10`, 59 families × 1 draw, 0 crashes, max CPU 597 s)
| Families | f2def alone (sweep1) | v10 |
|---|---|---|
| exact frames: k ≤ 4, pair insertions ≤ 2, layered, public frame examples, merged, Cup 2 frame B | 100 | 100 |
| sparse 2–24 gates | 0–100 (8–24 g: 0) | **90–100** |
| `open_example` (default and 72/24) | 0 | **100** |
| generic 72g with only 4 entanglers | 0 | 63.8 |
| generic small-angle 72g | 0 | 30.2 |
| generic 72g/12e, brick, angles 2.5–3.0; frames k ≥ 6; near-Clifford frames; pair_ins ≥ 6 | 0 | 0 |
| **mean** | 36.7 | **62.3** |

**Caveat for honest reporting:** the unbanded `generic` family at seed 11 (36/12 and 72/24) scored 100 / 99.8 in 12–19 s because `lab.generic` replays `open_example`'s RNG sequence, so the generator source recognised them as `open_example` seeds. That demonstrates seed recovery, not unknown-architecture decoding. The banded families used everywhere else, and our submitted attacks, have no such correspondence.

**Final attacks vs v10:** 0.0 on all 6 encounters (both templates × 3 draws).

## 16. Organizers' Championship Submission Check, edited templates, and v11 (Thursday night)

### 16.1 The organizers' check notebook, run on our ZIP
The organizers shared an optional notebook (`Quantum_Duel_Championship_Prep.ipynb`) that separates **execution ≠ inference ≠ synthesis ≠ submitted correction**. It reports the patch actually recorded at each checkpoint, its legality and size, and that checkpoint's ε and points.

- **SDK:** the notebook's 34 embedded SDK files are byte-identical to our `_quantum_duel_sdk_0_7_2` (`diff -r`), so no rule, limit or scoring change.
- **Headless runner:** `dev/open8/champ_check.py` executes the notebook's own helper cells (package check, SDK smoke runner, own-attack runner, evidence review) without Jupyter. Evidence goes to `dev/open8/data/champ_checks/`.
- **Static check:** `STATIC_VALIDATED`; sha256 `0a1cda0d…aaccbeec` = `submission_v10_validated.zip`.
- **Every case:** ZIP `MATCH`, Rules `MATCH`, all patches legal:

| Case (seed) | cp1 | cp2 | cp3 | Recorded patch |
|---|---|---|---|---|
| local (41) | 100 (ε 1.9e-6) | 100 | 100 | 1 gate |
| zz (41) | 100 | 100 | 100 | 8 g / 2 e |
| mixed (41) | 100 (3.4e-4) | 100 | 100 | 18 g / 6 e |
| multilayer (41) | 92.1 (1.44e-3) | 100 | 100 | 72 g / 24 e |
| frame 4/41 (41, 42, 43) | 100 (4.6–8.0e-5) | 100 (**6.0–7.4e-4**) | 100 (3.9–4.6e-4) | 82–85 g / 32–36 e |
| own attack 1, brick (73) | 0 (ε 0.99999) | 0 | 0 | 40–64 g / 24 e |
| own attack 2, all-pairs (73) | 0 (ε 0.9999) | 0 | 0 | 56–57 g / 24 e |

- **Our patch path:**
  - The final pool filter `legal()` calls the same `validate(patch, **rules.validation_kwargs())` that `LocalSession.patch` applies.
  - Each stage does an explicit `submit_patch` then `close_checkpoint`, using 3 of the 60 allowed patch updates.
  - Models are circuits and the patch is their exact inverse, so there is no separate synthesis step in which an estimate can be lost. The one way it was lost before, unbounded angles, was fixed in v4 (§13.1).
  - 99/99 v10 encounters completed, with no rejection lines in any log.
- `lab.py` now records `cp_gates` / `cp_ents` per checkpoint, so a zero score can be read as an empty patch or a wrong patch.

### 16.2 What the notebook exposed: refitting exact Clifford angles injected shot noise
In the frame rows above, ε got **~10× worse after checkpoint 1**. Our own logs show this is systematic:

- Of 30 logged encounters where `f2def:LEGAL_COMPILED` won checkpoint 1, `refit:previous` won checkpoint 2 in 27, and checkpoint 2 was worse in 23.
- On a Clifford-only frame, ε went from 4e-15 to 3.4e-4.
- **Cause:**
  - At checkpoints 2 and 3 the stale-model refit re-estimated all ~82 angles, including the ±π/2 Clifford ones f2def knows exactly. That adds shot noise to every parameter: ε ≈ k/(c·N), 5e-4 at 64k shots and 3.5e-4 at 96k, which matches the observed 1/N scaling.
  - BIC could not reject it, because `Model` counted k = 82 for both the exact model and its refit, so the refit won on raw NLL.
- **No points were lost yet** (worst case 7.9e-4 < 1e-3), but a larger frame or a team's reused Cup 2 frame would sit at the threshold.
- **Fix (v11):**
  - `free_mask` treats angles that are multiples of π/2 as exact.
  - `model_from_patch` counts only the free angles in k.
  - The stale refit holds the exact angles fixed with equal L-BFGS-B bounds and re-estimates only the insertion angles.
  - Models with continuous angles (pursuit, template, generator, subsys) are unaffected.

### 16.3 p1 vs p3 (closed-form scoring, top-10, insert-only) — `data/p3.log`, generic band attacks, seeds 21–23

| Shots / budget | Target | p1 | p3 |
|---|---|---|---|
| 64k / 200 s | 36 g | 58, 0, 0 | 100, 65, 0 |
| 64k / 200 s | 48 g | 0, 0, 0 | 0, 0, 0 (ε 0.35–0.48 vs ~1.0) |
| 96k / 400 s | 36 g | **100, 100, 100** | 100, 72.5, 28 |
| 96k / 400 s | 48 g | 3.7, 0, 0 | **100**, 0, 0 |

- **Totals:** at 96k shots the two score the same overall (304 vs 301 of 600), but they fail on different targets.
- **p3's behaviour:** it stops on BIC after 160–300 s, having overshot the gate count (41–64 found for 36/48). Its misses are near-misses: ε 0.004–0.18, where p1's misses are at ~1.0.

### 16.4 Polishing p3's near-misses — `data/polish.log`
Each run used p3, then a 1000-iteration refit, then stepwise deletion, then a warm p2 continuation, all inside p1's budget.

- The refit and deletion changed nothing: deletion removed 3–12 surplus gates at no gain.
- The **warm p2 continuation** rescued 36 g/22 at 96k (28 → 100) and moved two 48 g cases from ε 0.17 to 0.04–0.06.
- Of the 12 runs, 4 score 100, the same as raw p3.
- **Verdict:** not worth the CPU in the defender, so p1 stays in hard mode.

### 16.5 Edited public templates — `edits_v10` (v10, 16 encounters)
These model a team that copies a notebook default and edits a few gates.

| Edit | Edited `mixed` (18 g) | Edited `multilayer` (72 g) |
|---|---|---|
| delete 4 | 100, 80.7 | 99.6, 98.2 |
| insert 4 | 100, 100 | not a legal attack (76 > 72 gates; spec error) |
| change 3 axes | 100, 33.3 | **0, 0** |
| retarget 2 | 100, 100 | **0, 0** |

On the failures, the template screen ranked `bank:multilayer` first at separation 0.83–0.96, which v10 treats as no match. The fresh pursuit cannot reach 72 gates.

### 16.6 Template-seeded repair (offline, `t_seeded.py`, `data/seeded.log`)
Method: fit the nearest template (bounded, then free), then run p2 insert/delete from it with the **patch-sized caps (108 / 36)**, so it can insert past 72 gates before deleting.

| Case | Separation | Template fit | Seeded p2 at 32k (CPU) | Then at 96k |
|---|---|---|---|---|
| multilayer, 3 axes, d1 | 0.84 | ε 0.51 | 85.0 (62 s; +7 −2 gates) | **100** |
| multilayer, 3 axes, d2 | 0.82 | ε 0.50 | 96.9 (37 s) | **100** |
| multilayer, retarget 2, d1 | 0.94 | ε 0.99 | 100 (69 s) | **100** |
| multilayer, retarget 2, d2 | 0.95 | ε 0.86 | 96.0 (76 s) | **100** |
| mixed, 3 axes, d1 | 1.00 | ε 0.82 | 100 (11 s) | 100 |
| controls: multilayer, delete 4 | 0.50 / 0.72 | 91.7 / 96.5 | 0 insertions (unchanged) | 100 |

- **The trigger:**
  - Separation alone cannot trigger this, because near-matches (0.82–0.96, and 1.00 for the short `mixed`) overlap unrelated attacks (0.91–1.00).
  - The **identity of the best-screened template** does separate them. Across every logged encounter (90+), an unrelated attack's best screen was always a generator instance (`open_example:S:…`, `multilayer:S`), never a named public example.
- **Rule:** seed when there is no match, the template has ≥ 12 gates, and either the best template is a named public example (`bank:`, `notebook:`, `frame2:`, `frame4:`, `merged:`) or separation < 0.9.

### 16.7 v11 changes
1. Clifford-aware parameter count and refit (§16.2).
2. **Template-seeded repair:**
   - `TemplateSource.near` and `fit_near`; the pursuit engine `seed` (p2 with the patch-sized caps, 108 / 36); `PursuitSource.seed`.
   - Checkpoint 1 seeds for 90 s (`SEED_BUDGET`) when there is a near match and the data are still unexplained.
   - The fresh p2 run is skipped if the seeded model explains the data.
   - A seeded lineage keeps its warm engine, and in hard mode checkpoint 3 continues the repair rather than starting a fresh p1.
3. Unchanged for unrelated attacks: without a near match the code path is v10's.

### 16.8 v11 regression (`v11`: the v10 set + the 7 legal edits + both final attacks, draws 1–2, 52 encounters, 0 crashes, max CPU 579 s)
**Paired mean: v11 81.1 vs v10 72.9.**

| Group | v10 | v11 | Note |
|---|---|---|---|
| multilayer, 3 axes changed | 0, 0 | **97.0, 97.5** | seeded; 74–75-gate patches; 71–77 s CPU |
| multilayer, retarget 2 | 0, 0 | **89.7, 96.8** | seeded; d1 at ε 1–2e-3 on all three checkpoints |
| mixed, 3 axes changed | 33.3, 100 | **83.1**, 100 | d1 stuck at ε 2e-3 (a misfit the z test cannot see, §15.3) |
| other edits (delete 4, insert 4 into mixed, retarget 2 of mixed) | 80.7–100 | same | — |
| frames (k2, k4, frame 4/41, Cup 2 frame A, merged) | 100 | 100 | **cp2–cp3 ε now 1e-9 to 4e-5** (was 3–8e-4) |
| public / generator examples, 3-qubit 72 g, sparse, 24 g | 97.4–100 | same | — |
| generic 36 g / 48 g (hard mode) | 0–75.9 | same ±4.5 | 36 g/21 d2 fell 6.8 → 2.3; runs have CPU-time deadlines, and both scores are failures |
| our final attacks | 0 × 4 | 0 × 4 | — |

### 16.9 Broad validation of v11 (`broad_v11`, the same 59 families × draw 3 as §15.8, 0 crashes, max CPU 583 s)
- **Mean:** v11 62.7 vs v10 62.3. No family is more than 1 point worse.
- **Better:** generic 72 g with 4 entanglers 63.8 → 86.2, small-angle 72 g 30.2 → 32.2, and 72 g with angles 2.5–3.0 went 0 → 3.6. These come from CPU-time variation, not from a new mechanism.
- **Frame checkpoint-2 ε** (all 100-point frames): median **5.1e-4 → 5.9e-6**, max **7.1e-4 → 4.4e-5**. Frames now have a ~20× margin under the 1e-3 threshold, where they previously had ~1.4×.
- **The near-match trigger fired once,** on a near-Clifford frame (k2, width 0.12) whose best screen was `frame4:37` at separation 1.00. The frame4 name made it count as a near match. It scored 0 before and after, so the only cost was 90 s of stage-1 CPU.

### 16.10 v11 submission (validated)
`dev/build.py --profile quantum-duel-8q-open-0.7.1 --version v11 --files main.py opendef.py sim8.py pursuit.py subsys.py f2def.py tlib.json --cases local zz mixed multilayer frame --timeout 900`

- **VALIDATED_LOCALLY, valid_for_upload: True,** 228,053 bytes zipped / 839,095 expanded. Attack qualifier: VALIDATED.
- **Smoke, now including the public `frame` case:** local 100, zz 100, mixed 100, multilayer 97.37, frame 100 (checkpoint-2 ε 3.5e-6, where v10 was 6.0e-4).
- **Notebook package check:** `STATIC_VALIDATED`.
- `submission.zip` = `submission_v11_validated.zip`, sha256 `a43340a38ddeef2369ceb4e1443e454b04f75bef9781adb9b7a043735e4127d3`. The v10 fallback is kept as `submission_v10_validated.zip`.

## 17. Overnight attack generation (Friday 01:12–03:59) and v12

### 17.1 What ran
A second Claude session built and queued this run (`gen_overnight.py`, `queue_overnight.sh`, `overnight_report.py`) against a **frozen copy of the v11 ZIP**. The sets:

- **`fuzz`:** 116 legal attacks. 100 are random over the whole grammar: 1–72 gates, 2/3/5/8-qubit supports, random, line, star and few-pair topologies, and uniform, band, small, Clifford, near-Clifford and ±π-boundary angles. 16 are edge cases.
- **`edits2`:** 126 edited public templates, from 15 edit types × 2 variants on `mixed`, `multilayer`, `ex23_36` (the SDK default `open_example(rules)`), `ex23_72` and `frame`.
- **`v11cap400`:** the v11 set with `ENCOUNTER_CAP` 400.
- **Offline p1/p2/p3** on seeds 24–29.

**Review:**
- The design is sound: a frozen ZIP, decision-relevant stages first, SDK legality checked, 8 workers as in every earlier run, and `caffeinate` to keep the Mac awake.
- Training a learned "regression model" instead was rejected, for three reasons:
  - The estimator is already maximum likelihood; failures come from the structure search.
  - There is no training data that matches opponents' private architectures.
  - The runner has no deep-learning stack and the deadline is the same day.

### 17.2 Results (v11)
| Set | n | Mean | Zeros | Crashes | Max CPU |
|---|---|---|---|---|---|
| fuzz | 116 | 85.8 | 5 | 0 | 581 s |
| edits2 | 126 | 56.6 | 49 | 0 | **679 s** (over the cap) |
| v11 set at cap 400 | 52 | 78.5 (vs 81.1 at 600) | 10 | 0 | 379 s |

- **fuzz:**
  - ≤ 11 gates: 97.6.
  - 12–23 gates: 92.8.
  - The zeros ≤ 24 gates are all-entangler circuits (`ents_only_24`, a 22-entangler fuzz case). Also zero: `star_48`.
  - Weak spots: near-identity 72 g 41.3, ±π-boundary 72 g 47.4.
- **cap 400:**
  - The only losses are the hard-mode 36/48 g checkpoint-3 successes (4 × 33.3 → 0).
  - Everything else is identical.
  - This is the price of timeout insurance if the official limit turns out lower.
- **Offline, 96k shots / 400 s, seeds 24–29:**

  | Target | p1 | p2 | p3 |
  |---|---|---|---|
  | 36 g | 53.5 (2/6 solved) | **87.0 (4/6)** | 73.7 (3/6) |
  | 48 g | 0.1 (0/6) | 15.7 (1/6) | 22.6 (1/6) |

### 17.3 Diagnosis of the edits2 zeros (from the per-encounter logs)
1. **Rejected shortlist fits were never repaired (~18 zeros).**
   - Examples: multilayer axis1/axis3/retarget1/retarget2, and ex23_36 axis1/retarget/insert4/combo.
   - In each, the screen found the right template (separation 0.47–0.80), and the fit was then rejected at z = 17–117 because a few edited gates misfit.
   - The rejected key was dropped from the shortlist, but v11 seeded only screen-level near matches (`separation ≥ 0.8`). These fell through to a fresh pursuit, which cannot reach 36–72 gates.
2. **The SDK default was not "named."**
   - `open_example(rules)` defaults to seed 23 / 36 g / 12 e, so `open_example:23:*` is the most likely starting point for a team's own attack.
   - It is a generator-instance key, so near matches at separation 0.91–0.98 did not trigger (ex23_72 axis3/axis6/delete12/retarget4/combo).
3. **Whole-circuit edits:**
   - With every angle **negated**, or the gate order **reversed**, the attack screens as unrelated (best key `multilayer:91`/`multilayer:27`, separation 1.00).
   - Angle shifts of +0.4 rad on the 72-gate default also screen as unrelated (not fixed).
4. **False near-match trigger:** `frame4:37` was the best screen (separation ~1.00) of several unrelated attacks and triggered 90 s of useless seeding.
5. **CPU over the cap:**
   - An edited frame that still looks frame-like (reverse, retarget2) keeps f2def running at every checkpoint.
   - f2def's own synthesis, fit and fallback budgets (150/100/60 + 45 + 20 s) are not under `ENCOUNTER_CAP`.
   - Stage CPU was 337 → 554 → 679 s.

### 17.4 v12 changes
1. **A rejected template fit becomes a repair seed.** A shortlisted template whose fit is rejected (z ≥ 10) becomes `near`, and its fitted angles seed the insert/delete repair directly. Seeding may now happen once at checkpoint 1 or 2.
2. **The list of named public templates:** `bank:mixed`, `bank:multilayer`, `notebook:`, `open_example:23:*` (the SDK default), `open_example:31:*`, `multilayer:37`, plus the variants below. Frame templates are removed.
3. **Screen variants** `neg:`, `rev:` and `inv:` (all angles negated, reversed order, and both = the exact inverse) for 6 public templates: 18 extra screen entries.
4. **f2def CPU guard at checkpoints 2–3.**
   - f2def's synthesis/fit/fallback budgets are scaled to `ENCOUNTER_CAP − used − reserve` (reserve 120 s at checkpoint 2, 30 s at checkpoint 3).
   - f2def is skipped when less than 30 s would be left.
   - They are reset every stage, because module globals persist across encounters in one process.
   - On normal frames nothing changes, because checkpoint 2 starts with ≥ 280 s available.
- **Smoke:**

  | Case | v11 | v12 |
  |---|---|---|
  | ex23_36 negate | 0 | **98.5** (11 s) |
  | multilayer axis1 | 0 | **97.5** (58 s) |
5. **Hard-mode checkpoint-3 engine p1 → p2.** Pairing all 96k / 400 s offline runs (seeds 21–29):

   | Target | p1 solved (sum) | p2 solved (sum) | p3 solved (sum) |
   |---|---|---|---|
   | 36 g | 5/9 (621) | **6/9 (790)** | 4/9 (643) |
   | 48 g | 0/9 (4) | **3/9 (294)** | 2/9 (235) |

   The v5 choice of p1 rested on seeds 21–23 only, where p1 went 3/3 and p2 had not been run at 96k.

### 17.5 v12 regression (edits2 / v11 set / fuzz, from a frozen snapshot) and the v12.1 corrections
| Set | v11 | v12 | Zeros v11 → v12 | Max CPU v11 → v12 |
|---|---|---|---|---|
| edits2 (126) | 56.6 | **84.2** | 49 → 15 | 679 → **589** |
| v11 set (52) | 81.1 | 79.4 | 5 → 7 | 579 → 587 |
| fuzz (116) | 85.8 | 85.6 | 5 → 5 | 581 → 576 |

- **Rejected fit → seed, and the SDK default as a named template:**
  - These turned 0 into 92–100 on every multilayer axis/retarget edit and on most `open_example:23` (36 g and 72 g) edits.
  - Every trigger from a named, non-variant template was a gain or neutral: 31 gains, 0 losses.
- **The p2 hard-mode engine failed inside the defender.**
  - On the same attacks, v11's p1 made 4 checkpoint-3 successes (36 g/21 d1, 36 g/22 d1 + d2, 48 g/21 d2); v12's p2 kept 1.
  - Example, 36 g/21: p1 reached NLL 56,441 at 96k (at the truth); p2 got stuck at 158,612.
  - The offline advantage (§17.4 item 5) did not transfer to the defender's data (f2def probe + 540 random settings at uneven shot counts).
  - **Reverted to p1.**
- **The screen variants as repair seeds were harmful.**
  - `inv:bank:multilayer` at separation 0.98 seeded a wrong 68-gate model onto fuzz000, which won on BIC (99.1 → 59.3).
  - `neg:open_example:23:36:12` outranked the base on an axis edit (100 → 79.3).
- **v12.1:**
  - Variants are removed from the named list; they still serve exact matches below 0.8.
  - `fit_near` fits both angle orientations (centres and negated centres) and keeps the lower NLL.
  - The checkpoint-3 engine goes back to p1.
- Rerun as `*_v12b`.

### 17.6 v12.1 regression (`*_v12b`), and a bug found in it
| Set | v11 | v12.1 | Better / worse (> 5 pts) | Max CPU |
|---|---|---|---|---|
| edits2 (126) | 56.6 | **84.8** | 39 / 4 | 579 s |
| v11 set (52) | 81.1 | 80.4 | 0 / 1 (hard-mode 36 g/22 d1, 33.3 → 0) | 580 s |
| fuzz (116) | 85.8 | 85.5 | 3 / 5 | 576 s |

- **Two fuzz cases lost exactly the same points in v12 and v12.1** (fuzz007 92.2 → 86.2, fuzz060 93.5 → 88.3). Identical numbers mean a systematic change, not timing noise.
- **Their logs show `f2def:EXCEPTION_KEPT_LAST` at checkpoints 2–3:** a `TypeError` in f2def's shot allocation.
- **Cause: variable shadowing.**
  - The v12 CPU guard stored f2def's CPU allowance in `avail`, which in `run_defender` already holds the block's *shots*.
  - It then assigned `bridge.remaining = avail`, so f2def got a float of CPU seconds as its shot budget and raised.
- **Effect:**
  - 70 of 294 encounters (all where f2def runs at checkpoint 2 or 3); never in v11.
  - f2def's share of the block went to the random panel instead, and f2def contributed no new model.
  - Frames solved at checkpoint 1 kept 100, via the carried-over model.
  - **The CPU guard was therefore never actually exercised,** and the lower CPU on edited frames came from f2def failing early.
- **Fix:** the guard uses its own variable (`f2_cpu`).
- **Rerun:** exactly the 70 affected encounters (`fix_v12c`). The other 224 never entered the guard branch (`run_f2 and stage > 1`), so their v12.1 results stand.

### 17.7 After the fix (`fix_v12c`, 70 encounters): v12.1 vs v11 on all 294
| Set | v11 | v12.1 + fix | Zeros | Better / worse (> 5 pts) |
|---|---|---|---|---|
| edits2 (126) | 56.6 | **85.4** | 49 → 14 | 39 / 3 |
| v11 set (52) | 81.1 | 80.4 | 5 → 6 | 0 / 1 |
| fuzz (116) | 85.8 | 85.7 | 5 → 5 | 2 / 2 |
| **all 294** | **72.5** | **84.6** | | |

- **Health:** 0 f2def exceptions, 0 crashes, max CPU 593 s. Nothing over the cap, including the edited frames that reached 679 s in v11 (now 593 s).
- **The remaining losses:**
  - Hard-mode 36 g/22 d1 (33.3 → 0), and two deep fuzz partials (−6 and −13). These are all CPU-deadline pursuit runs with run-to-run variance.
  - **multilayer retarget4 v1, 93.4 → 37.5 — systematic.**
    - The two-orientation `fit_near` started the repair from a different basin.
    - The repaired model landed at z = 3.5, just above `GOOD_Z` = 3, which set hard mode, and the warm-only continuation stayed at ε ≈ 0.02.
    - v12, with a single orientation, reached z = 0.5 and 93.4.
- **v12.2:** the second (negated) start is used only when the near key is itself a `neg:`/`inv:` variant, the only case where the orientation is ambiguous. The 41 encounters seeded from a screen-level near match were rerun (`near_v12d`); rejected-fit seeds do not call the fit.

### 17.8 v12.2 final (`near_v12d` merged): all 294 encounters vs v11
| Set | v11 | v12.2 | Zeros | Better / worse (> 5 pts) | Max CPU |
|---|---|---|---|---|---|
| edits2 (126) | 56.6 | **85.9** | 49 → 14 | 39 / 1 | 593 s |
| v11 set (52) | 81.1 | 80.4 | 5 → 6 | 0 / 1 | 580 s |
| fuzz (116) | 85.8 | 85.7 | 5 → 5 | 2 / 2 | 576 s |
| **all 294** | **72.5** | **84.8** | | | |

- **multilayer retarget4 v1 is back to 93.4**; axis6 v2 went 91.6 → 96.6.
- **The remaining losses:**
  - Edited frame reverse0 v2 fell 14.4 → 0. v11 earned those points at 675 s CPU, over our cap, so this is the intended trade.
  - Three CPU-deadline pursuit runs: hard-mode 36 g/22 d1 and two deep fuzz partials. For these cases the code path is v11's apart from 18 more screen entries.
- **v12 components retained:**
  - A rejected template fit becomes a repair seed.
  - The SDK default `open_example:23:*` (plus seed 31 and multilayer 37) counts as a named template; frame templates are dropped.
  - neg/rev/inv screen variants, used for exact matches only.
  - The f2def CPU guard at checkpoints 2–3 (fixed).
  - The two-orientation near fit, used only for `neg:`/`inv:` keys.
  - p1 kept for hard mode.

### 17.9 v12 submission (validated)
`dev/build.py --profile quantum-duel-8q-open-0.7.1 --version v12 --files main.py opendef.py sim8.py pursuit.py subsys.py f2def.py tlib.json --cases local zz mixed multilayer frame --timeout 900`

- **VALIDATED_LOCALLY, valid_for_upload: True.** Attack qualifier VALIDATED; notebook package check `STATIC_VALIDATED`.
- **Smoke:** local 100, zz 100, mixed 100, multilayer 97.37, frame 100. Per-checkpoint ε is identical to v11, and on the frame f2def now runs at checkpoints 2–3 without the exception.
- **`submission.zip` = `submission_v12_validated.zip`,** sha256 `847c5e4a59e14d49e2d5bea74844db1617a0a725c769081e0fe5653b68edbefe`.
- **Fallbacks:** `submission_v11_validated.zip` (`a43340a3…`) and `submission_v10_validated.zip` (`0a1cda0d…`).

### 17.10 Is v12 overfit? A held-out test, with the decision rule stated before the results
There are two senses of overfitting.

- **Within an encounter (fitting shot noise):**
  - It is controlled by BIC (ln N per parameter), and every local score is the *true* process error against the hidden attack, so it cannot hide.
  - It was caught twice this way: the Clifford refit (§16.2) and the variant seed that won on BIC with a wrong 68-gate model (§17.5).
- **Design overfitting to our own test sets:**
  - v12's edits2 score (85.9) is **in-sample**: the fixes were designed from those 126 failures, and the fuzz and v11 sets shaped two further decisions.
  - What argues against it: the fixes are general mechanisms, and non-target sets did not degrade. But the size of the gain has not been measured out of sample.

**Held-out set** (`gen_holdout.py`, `specs_holdout.json`, 87 attacks, generated after v12 was frozen and not inspected):

| Part | Content |
|---|---|
| Edits (48) | 10 edit types on 5 bases. New instances (rng variant 3); edit types and sizes never used: qubit relabelling, angle scaling ×0.8, axis2, retarget3, delete8, insert2, shift2, widen5, swap3, combo2 (axis1 + retarget1 + delete2); one base not used in design (`open_example` seed 23 at 18 g). |
| Fuzz (39) | New fuzz seeds 100–139. |

- **Run:** both validated ZIPs (v11 `a43340a3…`, v12 `847c5e4a…`) run concurrently, 4 workers each, on draw 1 (`holdout_v11`, `holdout_v12`).
- **Decision rule, fixed in advance:**
  - Upload v12 if its paired held-out mean is ≥ v11's and there is no systematic class of losses, meaning no repeated loss pattern across an edit type or base.
  - Otherwise keep v11.
  - A single loss in a CPU-deadline pursuit run is noise, and repeated losses of one kind are not.

**Held-out result** (`holdout_v11` vs `holdout_v12`, 87 paired, 0 crashes, max CPU 588 / 577 s):

| Part | v11 | v12 | Zeros | v12 better / worse (> 5 pts) |
|---|---|---|---|---|
| Edits (48) | 79.8 | **90.1** | 9 → 4 | 5 / 0 |
| Fuzz (39) | 92.3 | 91.4 | 1 → 1 | 0 / 1 |
| **All (87)** | **85.4** | **90.7** | 10 → 5 | 5 / 1 |

- **All 5 gains came from edits never seen in design,** each through the repair path: ex23_36 retarget3 and combo2, ex23_72 delete8, multilayer axis2 and retarget3. All went from 0 to 97–100.
- **The one loss (fuzz124, 66.7 → 34.1):**
  - Identical checkpoint-1 decisions (same screen, generator top, and p2 model with NLL 30,159.9).
  - The runs diverged only in the 30 s CPU-deadline warm continuation at checkpoint 2, and v12's stage 1 ran slower (106 s vs 69 s of CPU for the same work).
  - v12 then solved it at checkpoint 3.
  - Per the rule, this is noise.
- **Decision (by the rule above): v12.** It is already `submission.zip`.
- **Shared gaps (both 0 or equal):**
  - A **qubit relabelling** of a public template (0/0 on ex23_36, multilayer, ex23_72): the screen compares targets literally.
  - ex23_72 retarget3 (0/0), ex23_18 widen5 (62/62), and mixed scale8 (82/82).
- **Caveat:** the held-out set was made with the same edit generator on mostly the same public bases, so it measures generalisation to new *instances and edit types*, not to arbitrary opponent designs.

## 18. The official runtime limit, oracle latency, and v13 (Friday afternoon)

### 18.1 What we learned
- **The organizers' answer (Discord, 16:34):** "each defender gets up to **600 seconds per encounter**." Whether this is wall-clock or CPU time was not stated.
  - A per-encounter limit enforced by the runner is almost surely wall-clock, like the SDK smoke runner's subprocess `timeout`.
  - The student notebook says a timeout "is not a completed synchronous defender run"; what happens to already-closed checkpoints is unknown.
- **LocalSession has no time limit at all** (no timing code in `local.py` or `smoke_worker.py`). The 600 s was **our own** `ENCOUNTER_CAP`, on CPU time (`time.process_time()`), which left *no* margin.
- **Rated runs reach the oracle over HTTP.**
  - `qduel_sdk/client.py`: "A fresh in-cluster TCP connection per request costs ~1 ms against ~30 ms of oracle work", with retries and back-off on 429/502/503/504.
  - Our encounters make 900–2,700 requests (the notebook check recorded 898–902 actions for continuous attacks, 2,327–2,728 for frames and warm-ups).
  - That is **~28–84 s of wall time invisible to a CPU-time cap.**
  - v12's heaviest encounters (~580–593 s CPU) would therefore finish at ~610–665 s wall on the server: **timeouts**. Templates, repairs and normal frames finish in 20–300 s and are safe.
- **Threads:** the defender code is single-threaded; numpy/scipy BLAS threads depend on the runner's environment.
  - Measured locally with the v12 ZIP (`thread_probe.py`), threads unset vs pinned to 1:

    | Case | Unset: wall / CPU (CPU ÷ wall) | Pinned: wall / CPU | Points |
    |---|---|---|---|
    | multilayer | 9.8 / 42.0 s (4.3) | 7.5 / 7.5 s | 97.37 both |
    | frame | 57 / 343 s (6.0, 11 native threads) | 149 / 141 s | 100 both |

  - With unpinned threads on a multi-core machine, a CPU-time cap runs out after 2.4–5.6× less work: safe for timeouts, but budget-bound runs lose search time.

### 18.2 v13 change
- **The encounter clock is `used()` = max(CPU, wall) since `run()` began.**
  - Wall catches oracle latency and scheduling; CPU catches inflation from unpinned BLAS threads.
  - It is used everywhere the encounter budget is checked: the f2def guard, generator and seeding gates, the pursuit plan, and the stale refit.
- **`ENCOUNTER_CAP` 600 → 540 s,** leaving 60 s for process start-up and imports before `run()`, the last step's overshoot, and oracle retries.
- **The stale-model refit** must now start before the cap; it previously ran up to cap + 60 s.
- **Test harness:** `lab.py` gained `LAB_LATENCY_MS` (each client call sleeps first: wall time without CPU time).
  - v13 was run on the v11 set (draws 1–2) + 4 edited frames + 2 long repairs at **35 ms** and 4 workers (`v13lat`).
  - Criterion: every `run()` under 600 s wall, and scores in line with v12.

### 18.3 v13 results and submission
- **Latency test (`v13lat`,** 58 encounters at 35 ms per client call and 4 workers):
  - 0 crashes. **Max `run()` wall 516 s** (none over 570 s); max CPU 477 s.
  - Paired against v12.2 without latency: **80.8 vs 76.5**, 4 better / 0 worse. Two edited frames went 0 → 100, and hard-mode 36 g/22 kept its checkpoint-3 successes.
  - The heaviest runs are our own attacks and an unsolvable edited frame, at 506–516 s wall.
- **Build:** `dev/build.py … --version v13 … --cases local zz mixed multilayer frame --timeout 900`
  - **VALIDATED_LOCALLY, valid_for_upload: True.**
  - Smoke: local 100, zz 100, mixed 100, multilayer 97.37, frame 100. Package check `STATIC_VALIDATED`.
- **`submission.zip` = `submission_v13_validated.zip`,** sha256 `ccdb04605b3b97eb0d097949224e120c764b49ca34a1b0e1c76d06efab72a21a`.
- **Fallbacks:** v12 `847c5e4a…`, v11 `a43340a3…`, v10 `0a1cda0d…`.

### 18.4 Cleaned submission (v13c)
- **Comments:**
  - The fully commented v13 sources are kept in `dev/open8/solution_v13_documented/`.
  - In the packaged sources, all 133 `#` comments were stripped mechanically (tokenizer-based), and every file's AST was verified identical before and after.
  - One docstring that cited this log was shortened.
- **Logging:**
  - `opendef.log()` is off by default; `OPENDEF_VERBOSE=1` turns it back on for lab diagnostics.
  - f2def's traceback and patch-rejection prints were removed.
  - Against the documented v13, pursuit, sim8, subsys and main are AST-identical ignoring docstrings, and f2def differs only by the removed prints.
- **Build:**
  - VALIDATED_LOCALLY, valid_for_upload: True. Smoke: local 100, zz 100, mixed 100, multilayer 97.37, frame 100. Package check `STATIC_VALIDATED`.
  - The packaged `opendef.py` contains no `#`.
- **`submission.zip` = `submission_v13c_validated.zip`,** sha256 `83f742975e97305b999e13a517e941802d63a7bbd12ea6afa1030ba0e67fb870`.
- **Fallbacks:** v13 `ccdb0460…` (same logic, commented), v12 `847c5e4a…`.
