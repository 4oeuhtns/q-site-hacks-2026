from __future__ import annotations

import time

import numpy as np

# time budget
TIME_BUDGET = 55.0
# reserve enough time to finish
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
    best_patch = ()
    best_model = None

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


def _legal_patch(as_gates, validate, rules, model):
    """convert attack model into a valid patch or None"""
    from duelkit.recovery4.quantum import inverse
    try:
        patch = as_gates(inverse(model))
        validate(patch, **rules.validation_kwargs())
        return patch
    except (ValueError, TypeError):
        return None


def _best_model(like, warm, previous, rules, budget):
    """collect every candidate we can afford, score them all by BIC, return the best"""
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

    # 1. persistent ensemble
    try:
        fitted = warm.fit(like)
        for candidate in fitted["candidates"].values():
            consider(inverse(candidate))
    except (ValueError, np.linalg.LinAlgError):
        pass

    # 2. refitted from scratch, warm start is bidirectional -> keep both and select best
    if time.time() - started < budget * 0.55:
        try:
            fresh = Ensemble(block=True, max_gates=rules.patch_max_gates,
                             max_entanglers=rules.patch_max_entanglers)
            for candidate in fresh.fit(like)["candidates"].values():
                consider(inverse(candidate))
        except (ValueError, np.linalg.LinAlgError):
            pass

    # 3. diversified restarts. DenseMLE.fit seeds np.random.default_rng(83177),
    #    change dense.x by hand, stop after several restarts without improvement
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
    """try removing, reordering and re-axising each gate; keep strict improvements."""
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
