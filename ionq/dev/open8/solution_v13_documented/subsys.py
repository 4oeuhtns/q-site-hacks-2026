"""Attacks confined to few qubits: exact support detection + overparameterized fit.

The oracle is noise-free apart from shot noise, so a qubit the attack never
touches returns its prepared eigenstate with certainty whenever it is read in
the preparation basis. Any disagreement marks the qubit as touched.

For a support of k <= 3 qubits the attack is a k-qubit unitary (63 parameters at
k = 3) whatever its depth. A fixed circuit with more parameters than that is
overparameterized: measured on exact 72-gate 3-qubit targets, every random start
reached eps ~ 1e-8. So it is fitted directly on the support's marginal counts.
"""
from __future__ import annotations

import math

import numpy as np

import sim8

AXIS = {'0': 'Z', '1': 'Z', '+': 'X', '-': 'X', '+i': 'Y', '-i': 'Y',
        'Z+': 'Z', 'Z-': 'Z', 'X+': 'X', 'X-': 'X', 'Y+': 'Y', 'Y-': 'Y'}
MINUS = {'1', '-', '-i', 'Z-', 'X-', 'Y-'}


def support(records, n, min_disagree=2):
    """Qubits the attack touches, from same-basis prep/readout disagreements."""
    bad = np.zeros(n)
    seen = np.zeros(n)
    idx = np.arange(1 << n)
    for r in records:
        c = np.asarray(r['counts'], float)
        for q in range(n):
            if AXIS[r['prep'][q]] != r['basis'][q]:
                continue
            bit = (idx >> (n - 1 - q)) & 1
            expect = 1 if r['prep'][q] in MINUS else 0
            bad[q] += c[bit != expect].sum()
            seen[q] += c.sum()
    return [q for q in range(n) if bad[q] >= min_disagree], seen


def marginal_data(records, sup, n):
    """FastData on the support qubits only (re-indexed 0..k-1)."""
    k = len(sup)
    idx = np.arange(1 << n)
    sub = np.zeros(1 << n, dtype=int)
    for j, q in enumerate(sup):
        sub |= ((idx >> (n - 1 - q)) & 1) << (k - 1 - j)
    agg = {}
    for r in records:
        key = (tuple(r['prep'][q] for q in sup), ''.join(r['basis'][q] for q in sup))
        c = np.bincount(sub, weights=np.asarray(r['counts'], float), minlength=1 << k)
        agg[key] = agg.get(key, 0) + c
    keys = list(agg)
    return sim8.FastData([kk[0] for kk in keys], [kk[1] for kk in keys], np.stack([agg[kk] for kk in keys]), k)


def ansatz(k):
    """Overparameterized k-qubit circuit (k <= 3), within the patch caps."""
    arch = []
    for q in range(k):
        arch += [('rz', (q,)), ('ry', (q,)), ('rz', (q,))]
    if k == 1:
        return arch
    pairs = [(0, 1)] if k == 2 else [(0, 1), (1, 2), (0, 2)]
    blocks = 6 if k == 2 else 16
    axes = ['zz', 'xx', 'yy']
    for b in range(blocks):
        a, c = pairs[b % len(pairs)]
        ax = axes[(b // len(pairs)) % 3] if k == 3 else axes[b % 3]
        arch += [('r' + ax, (a, c)), ('ry', (a,)), ('rz', (a,)), ('ry', (c,)), ('rz', (c,))]
    return arch


def fit(records, sup, n, deadline, restarts=4, seed=0, time=None):
    """Best (arch on the full register, angles) for the support, or None."""
    import time as _t
    data = marginal_data(records, sup, n)
    arch = ansatz(len(sup))
    ops = sim8.fast_ops(arch, len(sup))
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        if _t.process_time() > deadline and best is not None:
            break
        x0 = rng.uniform(-math.pi, math.pi, len(arch))
        x, v = sim8.fast_fit(ops, x0, data, maxiter=400)
        if best is None or v < best[1]:
            best = (x, v)
    full = [(name, tuple(sup[t] for t in ts)) for name, ts in arch]
    return full, best[0], best[1]
