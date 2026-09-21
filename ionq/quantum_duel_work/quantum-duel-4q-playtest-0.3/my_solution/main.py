"""Quantum Duel defender.

Design rationale (see ionq/findings.md for the measurements behind each choice):

* Measurement design is NOT the lever. The 180-setting random product panel is
  within ~2x of the A-optimal design, and optimal allocation is worth <=2 points
  on the encounter average. Closed-loop analysis circuits are measurably WORSE.
  So the measurement layer is deliberately the stock one, unchanged.

* The lever is architecture search under the 18-gate / 6-entangler patch cap.
  A fixed universal ansatz cannot invert a generic legal attack (eps 0.13-0.43
  even given the exact unitary), so the opponent's structure must be recovered.

* Three separate attempts to build a "better" single searcher all lost to stock
  somewhere, because stock's strength is its heterogeneous candidate pool under
  BIC selection, not any one search path. This defender therefore STRICTLY
  EXTENDS that pool and never replaces it:
      - the persistent (warm) ensemble, as stock runs it
      - a fresh ensemble refitted from scratch each checkpoint
      - the previous checkpoint's winning patch
      - local-search variants (delete / swap / axis-change), moves that stock's
        insert-only pursuit structurally cannot make
  and selects across the whole pool by BIC on the accumulated counts. That
  construction cannot score below stock except through selection noise.

Runtime target is ~75 s against a documented 180 s local default; every stage is
time-guarded and the defender always leaves a legal patch submitted.
"""
from __future__ import annotations

import time

import numpy as np

# Total wall-clock budget. The published local smoke default is 180 s and the
# server limit is undisclosed, so leave a wide margin.
TIME_BUDGET = 55.0
# Reserve enough to always finish submitting and closing the remaining stages.
STAGE_RESERVE = 6.0


def run(client, rules):
    from qduel_sdk.rules import OPEN8_RULESET

    if rules.version == OPEN8_RULESET:
        from duelkit.recovery8.open_starter import run_defender
        return run_defender(client, rules)
    if rules.qubits == 8:
        from duelkit.recovery8.adapter import run_defender
        return run_defender(client, rules)
    return _run_four_qubit(client, rules)


# --------------------------------------------------------------------------- 4q

def _run_four_qubit(client, rules):
    from duelkit.quantum import bitstrings, experiment_index, validate
    from duelkit.recovery4.adapter import as_gates
    from duelkit.recovery4.quantum import Likelihood, design, make_panel
    from duelkit.recovery4.recovery import Ensemble

    started = time.time()
    size = min(180, rules.max_settings, rules.block)
    panel = make_panel(4, size, 60231)
    ids = [experiment_index(e["prep"], "".join(e["basis"]), 4) for e in panel]
    kets, bras = design(panel)
    labels = bitstrings(4)
    counts = np.zeros((size, 16), dtype=np.int64)

    warm = Ensemble(block=True, max_gates=rules.patch_max_gates,
                    max_entanglers=rules.patch_max_entanglers)
    best_patch = ()          # best legal patch found so far, as G records
    best_model = None        # its attack model, as [{'pauli','angle'}, ...]

    for checkpoint in range(rules.checkpoints):
        base, extra = divmod(rules.block, size)
        for j, index in enumerate(ids):
            receipt = client.query(index, base + (j < extra), f"c{checkpoint}-s{j}")
            row = receipt["counts"]
            counts[j] += np.array([row[b] for b in labels], dtype=np.int64)

        like = Likelihood(kets, bras, counts)
        remaining = TIME_BUDGET - (time.time() - started)
        stage_budget = max(0.0, remaining - STAGE_RESERVE * (rules.checkpoints - checkpoint - 1))

        model = _best_model(like, warm, best_model, rules, stage_budget)
        if model is not None:
            patch = _legal_patch(as_gates, validate, rules, model)
            if patch is not None:
                best_patch, best_model = patch, model

        if best_patch:
            try:
                client.submit_patch(best_patch, note="pooled BIC selection")
            except (ValueError, TypeError):
                pass
        client.close_checkpoint()


BASE_PREP = {"X": "+", "Y": "+i", "Z": "0"}
FLIP_PREP = {"+": "-", "+i": "-i", "0": "1"}


def _parity_panel(n, size, seed):
    """Panel built as (base, one-qubit-flipped x n) families in a shared readout basis.

    These are still ordinary random product settings, so the dense MLE is
    unaffected, but the family structure is exactly what the recovery8 parity
    learner needs. Both candidate generators then read the SAME counts instead
    of competing for the 180-setting cap.
    """
    if size % (n + 1):
        return None
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    while len(out) < size:
        axes = rng.choice(list("XYZ"), n)
        basis = list(rng.choice(list("XYZ"), n))
        prep = [BASE_PREP[a] for a in axes]
        family = [(list(prep), basis)]
        for q in range(n):
            flipped = list(prep)
            flipped[q] = FLIP_PREP[flipped[q]]
            family.append((flipped, basis))
        key = (tuple(prep), tuple(basis))
        if key in seen:
            continue
        seen.add(key)
        out.extend({"prep": p, "basis": b} for p, b in family)
    return out[:size]


def _frame_candidate(panel, counts, n, rules):
    """Algebraic frame model from surviving Pauli relations, read off the panel.

    This is the recovery8 mechanism, which is completely different from the
    dense-relax-then-project pipeline: it identifies an exactly-Clifford frame
    from parity structure rather than by descending a likelihood. It wins big on
    exactly-quarter-turn attacks and returns nothing otherwise, which is exactly
    what a pooled selector wants from an extra generator.
    """
    from duelkit.recovery8.clifford import complete_images, nullspace, parse, rank
    from duelkit.recovery8.clifford_pair import best_pair

    pairs, have = [], 0
    for start in range(0, len(panel) - n, n + 1):
        family = panel[start:start + n + 1]
        if len(family) != n + 1:
            break
        base = counts[start]
        support = np.flatnonzero(base)
        if len(support) < 1:
            continue
        v0 = int(support[0])
        parities = nullspace([int(v) ^ v0 for v in support[1:]], n)
        if not parities:
            continue
        axes = "".join({"+": "X", "+i": "Y", "0": "Z"}.get(a, "Z") for a in family[0]["prep"])
        out_axes = "".join(family[0]["basis"])
        for parity in parities:
            sign = (parity & v0).bit_count() % 2
            coeff, ok = [], True
            for row in counts[start + 1:start + n + 1]:
                values = {(int(v) & parity).bit_count() % 2 for v in np.flatnonzero(row)}
                if len(values) != 1:
                    ok = False
                    break
                coeff.append(next(iter(values)) ^ sign)
            if not ok:
                continue
            inp = "".join(a if bit else "I" for a, bit in zip(axes, coeff))
            oup = "".join(out_axes[q] if parity & (1 << (n - q - 1)) else "I" for q in range(n))
            x, z = parse(inp)
            u, v = parse(oup)
            keys = [a[0] | (a[1] << n) for a, _ in pairs]
            grown = rank(keys + [x | (z << n)])
            if grown > have:
                pairs.append(((x, z, 1), (u, v, (-1) ** sign)))
                have = grown
    if have != 2 * n:
        return None
    gates = best_pair(complete_images(pairs, n), n, trials=48)
    if len(gates) > rules.patch_max_gates:
        return None
    if sum(sum(a != "I" for a in g["pauli"]) > 1 for g in gates) > rules.patch_max_entanglers:
        return None
    from duelkit.recovery4.quantum import inverse
    return inverse(gates)


def _legal_patch(as_gates, validate, rules, model):
    """Convert a fitted attack model into a validated patch, or None."""
    from duelkit.recovery4.quantum import inverse
    try:
        patch = as_gates(inverse(model))
        validate(patch, **rules.validation_kwargs())
        return patch
    except (ValueError, TypeError):
        return None


def _best_model(like, warm, previous, rules, budget, frame=None):
    """Collect every candidate we can afford, score them all by BIC, return the best."""
    from duelkit.recovery4.quantum import inverse, unitary
    from duelkit.recovery4.recovery import Ensemble, deviance

    started = time.time()
    pool = []

    def consider(model):
        if not model or len(model) > rules.patch_max_gates:
            return
        ent = sum(sum(a != "I" for a in g["pauli"]) > 1 for g in model)
        if ent > rules.patch_max_entanglers:
            return
        try:
            value = deviance(like, unitary(model, 4), len(model))
        except (ValueError, np.linalg.LinAlgError):
            return
        if np.isfinite(value):
            pool.append((float(value), model))

    if previous is not None:
        consider(previous)
    if frame is not None:
        consider(frame)

    # 1. the persistent ensemble, exactly as the stock defender runs it
    try:
        fitted = warm.fit(like)
        for candidate in fitted["candidates"].values():
            consider(inverse(candidate))
    except (ValueError, np.linalg.LinAlgError):
        pass

    # 2. the same ensemble refitted from scratch; warm-starting is bidirectional,
    #    so keeping both and selecting takes the better of the two for free
    if time.time() - started < budget * 0.55:
        try:
            fresh = Ensemble(block=True, max_gates=rules.patch_max_gates,
                             max_entanglers=rules.patch_max_entanglers)
            for candidate in fresh.fit(like)["candidates"].values():
                consider(inverse(candidate))
        except (ValueError, np.linalg.LinAlgError):
            pass

    # 3. diversified restarts. DenseMLE.fit seeds np.random.default_rng(83177),
    #    a FIXED seed, so a plain second Ensemble reproduces the first exactly.
    #    Perturbing dense.x by hand is what actually buys an independent basin.
    #    Stop after several restarts without improvement: easy attacks are solved
    #    in the first second and burning the whole budget is pure timeout risk.
    rng = np.random.default_rng(17)
    stale = 0
    while time.time() - started < budget * 0.70 and stale < 10:
        best_before = min((v for v, _ in pool), default=np.inf)
        try:
            extra = Ensemble(block=True, max_gates=rules.patch_max_gates,
                             max_entanglers=rules.patch_max_entanglers)
            extra.dense.x = rng.normal(0.0, 0.06, len(extra.dense.ps))
            for candidate in extra.fit(like)["candidates"].values():
                consider(inverse(candidate))
        except Exception:
            break
        best_after = min((v for v, _ in pool), default=np.inf)
        stale = 0 if best_after < best_before - 1e-6 else stale + 1

    if not pool:
        return None
    pool.sort(key=lambda item: item[0])

    # 4. local search on the leaders: delete / swap / axis-change, none of which
    #    the stock insert-only pursuit can perform
    for _, model in list(pool[:2]):
        if time.time() - started >= budget:
            break
        improved = _local_search(like, model, budget - (time.time() - started))
        if improved is not None:
            consider(improved)

    pool.sort(key=lambda item: item[0])
    return pool[0][1]


def _local_search(like, model, budget):
    """Try removing, reordering and re-axising each gate; keep strict improvements."""
    from scipy.optimize import minimize
    from duelkit.recovery4.quantum import circuit_and_jac

    started = time.time()
    labels = [g["pauli"] for g in model]
    x = np.array([g["angle"] for g in model], dtype=float)

    def objective(values, seq):
        u, jac, *_ = circuit_and_jac(seq, values, 4)
        loss, grad = like(u)
        return loss, np.real(np.einsum("ij,kij->k", grad.conj(), jac, optimize=True))

    def refine(seq, values):
        return minimize(objective, values, args=(seq,), jac=True, method="L-BFGS-B",
                        bounds=[(-np.pi, np.pi)] * len(seq),
                        options={"maxiter": 120, "gtol": 2e-7, "ftol": 1e-11, "maxls": 25})

    try:
        best_value = float(refine(labels, x).fun)
    except (ValueError, np.linalg.LinAlgError):
        return None
    best = None

    for i in range(len(labels)):
        if time.time() - started >= budget:
            break
        moves = [(labels[:i] + labels[i + 1:], np.delete(x, i))]
        if i + 1 < len(labels):
            seq = list(labels)
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
            values = x.copy()
            values[i], values[i + 1] = values[i + 1], values[i]
            moves.append((seq, values))
        targets = [q for q, a in enumerate(labels[i]) if a != "I"]
        for axis in "XYZ":
            relabelled = "".join(axis if q in targets else "I" for q in range(4))
            if relabelled != labels[i]:
                seq = list(labels)
                seq[i] = relabelled
                moves.append((seq, x.copy()))
        for seq, values in moves:
            if not seq or time.time() - started >= budget:
                continue
            try:
                fit = refine(seq, values)
            except (ValueError, np.linalg.LinAlgError):
                continue
            if float(fit.fun) < best_value - 1e-9:
                best_value = float(fit.fun)
                best = [{"pauli": p, "angle": float(t)} for p, t in zip(seq, fit.x)]

    return best
