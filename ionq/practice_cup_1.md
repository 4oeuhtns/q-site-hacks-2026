# Quantum Duel — Practice Cup 1, explained

What we built, why, and how it works. Written to be readable without the
measurement detail; `findings.md` has the numbers and the experiments behind
every claim here.

---

## 1. The game in one page

Someone hides a small quantum circuit from you. Call it **A**, the *attack*. You
never see it. You are allowed to run experiments against it: prepare some input,
let the hidden circuit act, measure, and get back a list of bitstrings.

Your job is to build a second circuit **P**, the *patch*, that **undoes** A. You
win when running A and then P is the same as doing nothing at all.

The crucial word is "undoes". Not "produces the right answer for one input" —
**undoes, for every possible input**. The judge measures

```
error = 1 - |trace(P·A)|² / 256
```

which is zero only when P is the exact inverse of A. Score is 100 points at an
error of 0.001 or better, 0 points at 0.1 or worse, sliding logarithmically in
between. You get scored three times during a match, and the three scores are
averaged.

Every team is both attacker and defender. Your final score is:

```
(how well you undid their attack  +  100 - how well they undid yours) / 2
```

So a hard-to-undo attack is worth exactly as much as a good defender.

### The budget

| | |
|---|---|
| Qubits | 4 |
| Total measurements ("shots") | 12,000 |
| Scoring checkpoints | 3, so **4,000 shots each** |
| Distinct experiment setups | 180, for the entire match |
| Attack circuit | at most 12 gates, at most 4 two-qubit gates |
| Patch circuit | at most 18 gates, at most 6 two-qubit gates |

Two details that shape everything:

- **Shots do not roll over.** Each 4,000-shot block is used or lost. There is no
  "save it for later" — the split across checkpoints is fixed at 4000/4000/4000.
- **The patch cap is barely bigger than the attack cap.** 18 vs 12 gates. Hold
  that thought; section 3 explains why it's the whole game.

---

## 2. The one idea that matters

We expected this to be a *measurement* problem: choose clever observables, spend
shots where they count, squeeze precision out of a noisy budget.

It isn't. It's a **search** problem.

Here's the experiment that settled it. We took the defender and *told it the
answer's shape* — handed it the exact list of gates the attacker used, so all it
had to do was fit the rotation angles. On every attack we tried, it scored **100
points**. The 12,000 shots always contain enough information.

Then we let it keep the gate list but start the fitting from random angles.
It found the right answer roughly **one time in twelve**.

So the measurements are never the bottleneck. The bottleneck is **finding the
right circuit structure** among an astronomical number of wrong ones. Once you
know the structure, the data is more than enough. If you don't, no amount of data
helps.

Everything below follows from that.

### Why you can't dodge the search

The obvious escape: skip identifying the opponent's circuit and just fit one
big flexible circuit that can imitate anything.

We tested it with the strongest possible advantage — we gave the fitter the
attack's exact mathematical description, no measurement noise at all — and asked
it to approximate the inverse using a fixed 18-gate shape. Best result across
many tries: an error between **0.13 and 0.43**, when 0.1 already scores zero.
The true inverse scores 0.

That's the 18-vs-12 cap doing its work. The patch budget is too tight to absorb a
wrong guess about the structure. **You have to actually find the opponent's
circuit** (or something equally compact that does the same thing). Those six
spare gates cannot buy you out of a wrong answer.

### Things we confirmed are *not* worth effort

We measured these so nobody has to re-litigate them:

- **Shot allocation.** We computed the theoretically perfect allocation — one
  that already knows the attack, so it's an unreachable upper bound. It's worth
  about **2 points** on a match average, and nothing at all by the third
  checkpoint.
- **"Measure the leftover error."** The intuitive idea of applying your current
  best patch during the experiment and measuring what's left over is **worse**,
  by about 2×. The cleanest version of it is mathematically useless: when your
  patch is nearly right the outcome becomes nearly certain, and a certain outcome
  teaches you nothing.
- **Using fewer setups with more shots each.** Makes almost no difference.

---

## 3. How our defender works

File: `quantum_duel_work/quantum-duel-4q-playtest-0.3/my_solution/main.py`

### The shape of it

Three times per match, it does the same loop:

1. **Measure.** Run all 180 experiment setups, splitting that block's 4,000 shots
   evenly. Results accumulate, so the third checkpoint is fitting all 12,000.
2. **Guess, many ways.** Generate a pile of candidate circuits using several
   *different* methods.
3. **Pick the best guess**, by asking which one best explains the actual counts.
4. **Submit it**, then close the checkpoint.

Step 1 is deliberately unchanged from the stock SDK learner, because section 2
showed there's nothing to win there. All the work is in steps 2 and 3.

### Why a pile of guesses instead of one good method

This is the part we got wrong three times before getting it right.

Our first instinct was to build a smarter searcher and swap out the stock one.
We tried three: hundreds of random restarts, restarts wrapped around a smarter
initialisation, and a method tuned for the specific attack family that was
beating us. **Every one of them beat the stock learner on some attacks and lost
badly on others** — one of them turned a perfect 100 into a 0.

The reason: the stock learner isn't one algorithm. It's three different ones
running in parallel, with a rule for picking whichever explains the data best.
Its strength *is* the variety. Every time we replaced it with something
cleverer, we threw away the variety and lost the attacks that the discarded
methods happened to handle.

So the design rule became: **only ever add, never replace.**

Our defender keeps all three of the stock methods and adds more:

| Candidate source | What it contributes |
|---|---|
| The stock trio, carried forward between checkpoints | the baseline; sometimes momentum helps |
| The same trio, restarted from scratch each checkpoint | sometimes momentum *hurts*, and this catches it |
| Extra restarts from deliberately scattered starting points | escapes bad basins |
| Local edits to the best candidates | delete a gate, swap two, change an axis |
| The previous checkpoint's winner | never go backwards |

Then it scores every candidate against the real counts and submits the winner.

Because the stock candidates are always in the pile, **this construction can't do
worse than stock** except by narrowly mis-picking between two near-identical
options. That's a guarantee by design, not a hope — and the measurements bear it
out: 0 regressions across 17 test cases.

The "local edits" row deserves a note. The stock searcher builds a circuit by
*inserting* gates one at a time and can never remove or reorder one afterwards.
If it inserts a wrong gate early, that mistake is permanent for the rest of the
match. Our version can delete, reorder, and re-axis, which is a class of fix the
original structurally cannot make.

### Staying inside the limits

- **Time.** It budgets 55 seconds and stops early once extra attempts stop
  helping. Easy attacks finish in 5–8 seconds; only genuinely hard ones use the
  full budget. The documented limit is 180 seconds, so there's roughly 3× margin.
- **Never returns empty-handed.** A patch is submitted at every checkpoint, always.
  It's validated before submission, because an over-budget patch is silently
  rejected and you'd keep your older, worse one without being told.
- **Nothing can crash it.** Every candidate generator is wrapped. If one fails,
  the pile just has one fewer entry.

### What it achieves

Against the public practice attacks, compared to the stock learner:

| | stock | ours |
|---|---|---|
| Average score | 85.9 | **92.1** |
| Worst case | **0.0** | **56.5** |
| Crashes | 0 | 0 |

The worst case matters more than the average. Stock has attacks it scores
literally zero on; ours doesn't.

---

## 4. How our attacks work

File: `quantum_duel_work/quantum-duel-4q-playtest-0.3/attacks.json`

You submit exactly two attack templates. A template describes the gate layout and
an allowed *range* for each angle; the server rolls specific angles for each match.

### What doesn't work

We screened eight designs first. The two most intuitive ideas both failed:

- **"Scramble it as hard as possible."** Broad random angles on all 12 gates,
  output as close to random noise as we could make it. This was the **easiest**
  attack we built — opponents recovered 90 out of 100 against it. More scrambling
  means more *distinguishable* behaviour, which helps the person trying to figure
  you out. Exactly backwards from the intuition.
- **"Use every gate to make them estimate more numbers."** There's a hard ceiling
  here. With 12 gates maximum, the extra estimation error you can force is worth
  at most about 16 points at the first checkpoint and almost nothing later. **No
  attack built purely on making measurement harder can win.**

### What does work: a convincing wrong answer

The winning idea is not to hide information. It's to make sure there's a
**plausible wrong answer** sitting right where the opponent's search will look.

Our attacks set almost every rotation angle very close to a quarter turn (90°).
Why that specific value: a quarter-turn rotation has a special algebraic property
— it converts one type of measurement into another cleanly. The practical
consequence is that a circuit built from near-quarter-turns can be **rearranged
into many different-looking circuits that behave almost identically**.

So when the opponent's search goes looking for a compact circuit that explains
their data, it finds one. It's just the wrong one. And because the patch budget is
so tight (section 2), a wrong structure can't be patched up.

We confirmed this is really about quarter turns specifically:

| angles clustered near | opponent's score |
|---|---|
| **90° (a quarter turn)** | **0.0** |
| 180° (a half turn) | 87.1 |
| 57° (nothing special) | 84.1 |
| 126° (nothing special) | 89.6 |

Only the quarter turn does it. And the width of the range doesn't matter — a
narrow cluster and a wide one both scored 0.

We also verified it isn't merely that the search gives up early: the searcher
runs to its full gate budget and the very first gate it adds improves the fit by
about 5,000× more than needed to justify itself. It's working hard and confidently
producing the wrong answer.

### The mistake we caught late

Our first submitted pair had one template using angles set to *exactly* 90°.

That was a serious error. There's a second family of defender — one that works
algebraically rather than by trial and error — and exact quarter turns are
precisely what it's built to recognise. We ported that defender to 4 qubits to
check, and it **solved that template completely, in 0.2 seconds**.

The fix: use angles that *straddle* 90° without ever landing on it (our ranges run
about 83° to 97°). Then the algebraic shortcut finds nothing, while the
"convincing wrong answer" trap still works perfectly.

The lesson generalises: **near a quarter turn is a trap; exactly on it is a gift.**

### The final two

Both templates are 12 gates with 4 two-qubit gates, all angles ranged near but
never on a quarter turn. They differ in which qubit pairs are connected:

| | connections | vs stock defender | vs algebraic defender | vs our own defender |
|---|---|---|---|---|
| **cycle** | 0-1, 2-3, 1-2, 0-3 | **0.0** | **0.0** | 34.7 |
| **crossed** | 0-2, 1-3, 0-1, 2-3 | **0.0** | **0.0** | 11.0 |

Zero against both standard defender families, at every seed tested. `crossed` is
the stronger one.

The honest caveat: our own pooled defender does partially crack them, because
combining many methods is stronger than any single one. A team that builds
something similar will get partial credit against us.

---

## 5. What we got wrong

Kept deliberately, because the wrong turns were informative:

| We believed | Reality |
|---|---|
| Scrambling makes recovery hard | Backwards — it was our weakest attack |
| Measuring the residual error helps | ~2× worse; the clean version is mathematically useless |
| A flexible universal patch avoids the search | Impossible within the 18-gate cap |
| More restarts beat the stock searcher | Lost everywhere — we'd dropped a crucial stage |
| A cleverer single searcher beats the pool | Lost a perfect score on one attack |
| Aiming the search at quarter turns cracks the trap | No effect whatsoever |
| The trap works by stopping the search early | No — the search runs to completion, confidently wrong |
| Merged insertions hedge against algebraic defenders | Backwards for this round — they *enable* them |
| Two generators could share one measurement panel | Cost us 100 → 0 on two easy attacks; reverted |

Two of these were caught only because the test harness reports *crashed* and
*scored zero* as different things. A defender that dies and one that recovers
nothing both show 0.0, and only one is a bug.

---

## 6. Running it

```bash
cd ionq

# score our defender against the practice attacks, across all CPU cores
python3 dev/harness.py --defender quantum_duel_work/quantum-duel-4q-playtest-0.3/my_solution/main.py -j 0

# rebuild submission.zip without opening the notebook
python3 dev/build.py
```

`dev/build.py` packages the defender and both attacks, re-runs the same validator
the server uses, executes the result against practice attacks in fresh processes,
and refuses to produce an uploadable archive if anything fails.

Environment setup, including WSL2, is in `../SETUP.md`. Every measurement quoted
here is reproducible from `dev/experiments/`.

---

## 7. What we'd do next

1. **Narrow the search using connection structure.** We found you can reliably
   read off *which qubit pairs* an attack connects, straight from the data — 3 or
   4 out of 4 correct on every attack tested. We never used that to constrain the
   search. It's the most promising unexplored lead.
2. **A search that isn't greedy.** Everything that fails against our own attack
   type fails the same way: it commits to one gate at a time and gets led astray.
   Something that considers several gates jointly might break through.
3. **More draw seeds.** Most conclusions rest on 3–4 rolls per attack family.
   The large effects are unambiguous; the small ones may be noise.
4. **Confirm the time limit.** We assume the documented 180 seconds. If the real
   grading machine is much slower than a laptop, the 55-second budget should come
   down.
