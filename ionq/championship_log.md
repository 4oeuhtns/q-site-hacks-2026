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
| P1 | **Seed brute force of public generators** at runtime (`open_example`, `open_multilayer_example`, thousands of seeds, common sizes), screened on a few settings | E | **high**: decodes 72-gate attacks made by generators | none | low | built (v4), in regression |
| P2 | **Subspace/beam pursuit with rearrangement moves** (delete, move, swap, re-axis), several candidates per step, prune back | A | medium | **high**: target 36g/12e, then 48g/16e | medium | Pursuit2 built; mixed vs p1 (§13.2); both run in v4 |
| P3 | **Dictionary pruning by measured coupling graph** (restrict pair gates to pairs with evidence of coupling) | C1 | medium | medium: shrinks each step's search ~3× | low | deprioritized (§13.3) |
| P4 | **Residual pursuit:** at checkpoints 2–3 measure with the current best inverse as the analysis circuit, run pursuit on the shallower residual | D | low–medium | potentially high | medium–high (uses settings) | experiment only |
| P5 | **Relax-then-project:** overparameterized layered ansatz + L1, prune to legal gates | B | low | unknown | high | only if time |
| P6 | Clifford-doped extension of f2def past 4 insertions | F | low | low | high | deferred |
| P7 | Multithreaded kernels | G | depends on runner | medium | low | blocked: runner limits unknown |
| R1 | **Timeout safety:** f2def alone up to 514 CPU s; v3 generic cases reach ~690 s | Inversion | **critical** | — | low | open |
| R2 | Final attacks: verify vs v3, stock, and an overpowered adversary; build and validate the OPEN8 ZIP | — | **critical** | — | low | attacks verified; ZIP statically valid; final smoke pending |

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
