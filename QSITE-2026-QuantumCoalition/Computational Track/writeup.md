# Plan in Periods: Qubit Placement and Routing

*Aoeuhtns* · QSITE 2026 Quantum Coalition · Computational Track

## 1. Problem

The chip has 20 physical qubits but only 23 connections, 12% of all pairs. A two-qubit gate can run only on connected qubits, so the compiler must choose a starting **placement** (logical → physical qubit) and insert **SWAPs** that move qubits together. The gates must keep their order. The score is **SWAPs + ½ × depth** over six benchmarks, lower is better. The baseline (qubit *k* on position *k*, then greedy shortest-path SWAPs) scores **283.5**.

## 2. Approach

**Periods.** We cut the program into *periods*: the longest runs of consecutive gates that can execute with zero SWAPs under one placement. A period *fits* if its interaction graph is **subgraph-monomorphic** to the chip: every interacting pair can sit on a connection. We test this in two stages:

- **Instant impossibility proofs,** derived from the chip graph rather than hard-coded: a qubit with more partners than the chip's maximum degree (3); an odd cycle on a bipartite chip; a cycle shorter than the chip's girth (6).
- **A backtracking search** that seats qubits in breadth-first order, so each new qubit has at most 3 candidate positions next to a seated partner. It has a step budget and answers "fits" only when it has actually found a seating, so running out of budget can shorten a period but never make it wrong. (networkx's VF2 took 48 s to prove one non-fit and cannot be interrupted.)

Fitting is monotone, so each period's end is found by binary search.

**Choosing a seating.** All seatings that fit are free *inside* the period, but they differ in cost to reach and to leave. Branch and bound minimises the hops the period's qubits move from their current positions, plus the distance between the pairs of the next 20 gates (weights 1, 0.8, 0.8², …). The second term is a **look-ahead across periods**: it prefers seatings that leave the following gates close. Its weight of 1 counts a hop later the same as a hop now.

**Reshuffle or route.** For each period we build two candidates:
- **Reshuffle:** token-swap into the chosen seating (greedy distance-reducing SWAPs, with a spanning-tree fallback that always terminates), then run the period with zero SWAPs.
- **Route:** keep the seating and route gate by gate. Each SWAP moves one endpoint one hop closer, and a 20-gate look-ahead picks which SWAP to make.

We keep whichever has the lower **real score** for everything emitted so far, so a period is never worse than gate-by-gate routing.

**Starting seatings and refinement.** `solve()` tries two starting seatings:
- *centre-out:* the first period seated near the chip's centre with look-ahead, and other qubits next to their frequent partners;
- *label order:* qubit *k* on position *k*.

Each is refined with SABRE's forward-backward pass (route forwards, then route the reversed program from where qubits ended; their final positions become the new start) while the score improves, up to 5 rounds. Every candidate is validated, and the best is returned.

## 3. Results

| Benchmark | Baseline | Ours | SWAPs | Depth | Why it can't reach 0 SWAPs |
|---|---|---|---|---|---|
| ghz_star | 14.0 | 8.0 | 3 | 10 | a qubit has 7 partners; max degree 3 |
| chain_trotter | 15.0 | **4.5** | 0 | 9 | optimal |
| ladder_trotter | 35.5 | 9.0 | 5 | 8 | 4-cycles; chip girth 6 |
| qaoa_random | 39.0 | 15.0 | 9 | 12 | a qubit has 5 partners |
| dense_random | 122.0 | 49.5 | 34 | 31 | a qubit has 9 partners |
| vqe_layers | 58.0 | **3.0** | 0 | 6 | optimal |
| **Total** | **283.5** | **89.0** | | | 69% lower |

Two benchmarks reach zero SWAPs, which is optimal. For `ghz_star` the optimum is 6.5:
- At most 3 partners start adjacent to the hub.
- Each hub move reaches at most 2 new partners; each partner move reaches 1.
- So at least 2 SWAPs are needed, both involving the hub. That forces depth ≥ 9 and a score ≥ 6.5.

We score 8.0.

## 4. What each part contributes

Each row removes one component from the final system:

| Configuration | Benchmarks | Random suite* |
|---|---|---|
| Full system | **89.0** | **0.380** |
| − look-ahead across periods | 97.5 | 0.388 |
| − forward-backward refinement | 91.0 | 0.419 |
| − label-order starting seating | 91.0 | 0.398 |
| − period router (gate-by-gate only; period-based starting seating kept) | 90.5 | 0.392 |
| − all period machinery (SABRE-like: gate-by-gate + label order + refinement) | 128.0 | 0.465 |

\*Mean of our score ÷ baseline score over the 120 random programs of Section 5. Lower is better.

**The period machinery is worth 128.0 → 89.0 on the benchmarks and 0.465 → 0.380 on random programs.** Most of that comes from where qubits *start*: seating the first period so it fits, with look-ahead, and letting that seating drive the refinement. The period router's reshuffles add a smaller, consistent final step (90.5 → 89.0; 0.392 → 0.380). On the final seatings of the six benchmarks, the period router and the gate-by-gate router tie exactly.

## 5. Generality and validation

Every setting was chosen on a random suite, never on the benchmarks. The suite is 3 program styles (uniform, label-local, clustered) × 10 seeds (6–16 qubits, 10–64 gates) × 4 chips: the challenge chip, plus a 4×5 grid, a 20-qubit ring and a random 3-regular graph that the code was never tuned on.

| Chip | challenge | grid 4×5 | ring 20 | random 3-regular |
|---|---|---|---|---|
| Ours ÷ baseline (mean of 30) | 0.45 | 0.32 | 0.46 | 0.29 |

All 120 answers are valid, and none is worse than the baseline (worst 0.66×).

**Tests.** 50 tests cover every module, 800 random reshuffles on four chips, and a check that `solve` leaves its inputs unchanged. One test runs the self-contained `submission.py` (standard library + networkx only) with the rest of the repository hidden and checks it gives identical answers.

**Spec check.** It passes on 77 inputs, including an empty program, single-qubit-only programs, gappy labels and a 36-qubit chip.

**Speed.** Each benchmark solves in under 0.25 s.

## 6. Limitations and next steps

- **`ghz_star` scores 8.0 against an optimum of 6.5.** Reaching it means seating later partners around positions the hub has not reached yet, which is beyond our look-ahead.
- **Dense random programs gain little from reshuffling.** Each period of `dense_random` (8–10 gates) wants a very different seating: a reshuffle costs 10–31 SWAPs against 3–9 for routing, and routing wins 4 of 5 periods.

Next steps:
- **Periods that fit almost perfectly.** Allow a few pairs to be routed inside a period, so seatings can change less.
- **A stronger token swapper.** Reshuffles use 2.3× the SWAP lower bound on average (2.6× for large ones), and approximation algorithms with guarantees exist [3].

## References

[1] M. Y. Siraichi, V. F. dos Santos, C. Collange, F. M. Q. Pereira. *Qubit Allocation as a Combination of Subgraph Isomorphism and Token Swapping.* OOPSLA 2019. The "fit, then token-swap" idea we build on.
[2] G. Li, Y. Ding, Y. Xie. *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices* (SABRE). ASPLOS 2019. The source of the forward-backward refinement.
[3] T. Miltzow et al. *Approximation and Hardness of Token Swapping.* ESA 2016.
