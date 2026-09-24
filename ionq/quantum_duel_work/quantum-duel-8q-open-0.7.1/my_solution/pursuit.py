"""Greedy gate pursuit on the count likelihood, for attacks with unknown architecture.

One adjoint sweep gives d(NLL)/d(theta) at theta=0 for every dictionary rotation
at every insertion position. The strongest few are line-searched, the best is
inserted, and all angles are refitted. Stops on BIC, a gate cap, or a deadline.
"""
from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np

import sim8

try:
    import numba as _nb

    @_nb.njit(cache=False)
    def _cand_grads(lam, psi, rows_all, ph_all):
        C = rows_all.shape[0]
        S, D = psi.shape
        out = np.zeros(C)
        for c in range(C):
            acc = 0.0
            for j in range(S):
                for i in range(D):
                    v = lam[j, i].conjugate() * (-0.5j) * ph_all[c, i] * psi[j, rows_all[c, i]]
                    acc += v.real
            out[c] = 2.0 * acc
        return out
except Exception:  # pragma: no cover
    _cand_grads = None


def dictionary(n, pairs=None):
    d = [('r' + a, (q,)) for q in range(n) for a in 'xyz']
    for p in (pairs if pairs is not None else combinations(range(n), 2)):
        d += [('r' + a * 2, tuple(p)) for a in 'xyz']
    return d


class Pursuit:
    def __init__(self, n, max_gates=40, max_ents=24, pairs=None, top=10, penalty=None):
        self.n = n
        self.dict = dictionary(n, pairs)
        ops = sim8.fast_ops(self.dict, n)
        self.rows_all = np.ascontiguousarray(np.stack([o[0] for o in ops]))
        self.ph_all = np.ascontiguousarray(np.stack([o[1] for o in ops]))
        self.max_gates = max_gates
        self.max_ents = max_ents
        self.top = top
        self.penalty = penalty

    def position_grads(self, arch, angles, data):
        """grads[k, c]: derivative for inserting dict[c] after the first k gates."""
        ops = sim8.fast_ops(arch, self.n)
        st = sim8.fast_state(ops, angles, data)
        phi = st.copy()
        sim8._readout(phi, data.mats, data.n)
        p = np.maximum(phi.real ** 2 + phi.imag ** 2, 1e-12)
        lam = np.ascontiguousarray(-(data.counts / p) * phi)
        sim8._readout(lam, data.mats_adj, data.n)
        K = len(ops)
        G = np.zeros((K + 1, len(self.dict)))
        G[K] = _cand_grads(lam, st, self.rows_all, self.ph_all)
        for k in range(K - 1, -1, -1):
            rows, ph = ops[k]
            a = angles[k]
            c, s = math.cos(a / 2), math.sin(a / 2)
            sim8._rot(st, rows, ph, c, -s)
            sim8._rot(lam, rows, ph, c, -s)
            G[k] = _cand_grads(lam, st, self.rows_all, self.ph_all)
        return G

    def line_search(self, arch, angles, data, pos, cand):
        a2 = arch[:pos] + [self.dict[cand]] + arch[pos:]
        ops = sim8.fast_ops(a2, self.n)
        best = (math.inf, 0.0)
        for t in (-2.4, -1.6, -0.8, -0.3, 0.3, 0.8, 1.6, 2.4, math.pi):
            x = np.concatenate([angles[:pos], [t], angles[pos:]])
            v = sim8.fast_nll(ops, x, data)
            if v < best[0]:
                best = (v, t)
        # polish just the new angle
        from scipy.optimize import minimize_scalar
        def f(t):
            x = np.concatenate([angles[:pos], [t], angles[pos:]])
            return sim8.fast_nll(ops, x, data)
        t0 = best[1]
        r = minimize_scalar(f, bounds=(t0 - 0.8, t0 + 0.8), method='bounded', options=dict(maxiter=20))
        if r.fun < best[0]:
            best = (float(r.fun), float(r.x))
        return best, a2

    def run(self, data, deadline, arch=None, angles=None, trace=None):
        arch = list(arch or [])
        angles = np.asarray(angles if angles is not None else [], float)
        pen = self.penalty if self.penalty is not None else math.log(max(data.N, 2))
        cur = sim8.fast_nll(sim8.fast_ops(arch, self.n), angles, data) if arch else sim8.fast_nll([], [], data)
        while len(arch) < self.max_gates and time.process_time() < deadline:
            G = self.position_grads(arch, angles, data)
            ents = sum(len(t) == 2 for _, t in arch)
            if ents >= self.max_ents:
                for c, (nm, t) in enumerate(self.dict):
                    if len(t) == 2:
                        G[:, c] = 0
            flat = np.argsort(-np.abs(G).ravel())[: self.top]
            best = None
            for f in flat:
                if time.process_time() > deadline:
                    break
                pos, cand = divmod(int(f), len(self.dict))
                (v, t), a2 = self.line_search(arch, angles, data, pos, cand)
                if best is None or v < best[0]:
                    best = (v, t, pos, a2)
            if best is None:
                break
            v, t, pos, a2 = best
            if cur - v < pen:
                break
            x0 = np.concatenate([angles[:pos], [t], angles[pos:]])
            x, v2 = sim8.fast_fit(sim8.fast_ops(a2, self.n), x0, data, maxiter=150)
            arch, angles, cur = a2, x, min(v, v2)
            if trace is not None:
                trace.append((len(arch), round(cur, 1), arch[pos]))
        return arch, angles, cur


class Pursuit2(Pursuit):
    """Pursuit with exact closed-form insertion curves and backward elimination.

    Inserting R_P(t) at position p turns the final amplitudes into
    cos(t/2) A - i sin(t/2) B, with A the current output and B the output with
    P applied at p. One suffix simulation per candidate gives the likelihood at
    every angle, so many more candidates are scored exactly per step. After each
    refit, a gate whose removal costs less than the BIC penalty is deleted
    (stepwise in both directions), so an early wrong gate can be repaired.
    """

    def __init__(self, n, max_gates=60, max_ents=24, pairs=None, top=40, penalty=None,
                 refit_iter=60, prune=True):
        super().__init__(n, max_gates, max_ents, pairs, top, penalty)
        self.refit_iter = refit_iter
        self.prune = prune
        self.grid = np.linspace(-np.pi, np.pi, 73)[:-1]
        self.stats = dict(inserted=0, deleted=0, steps=0)

    @staticmethod
    def _amps(ops, angles, data):
        st = sim8.fast_state(ops, angles, data)
        sim8._readout(st, data.mats, data.n)
        return st

    def _curve_best(self, a, b, r, cnt):
        best_v, best_t = math.inf, 0.0
        for t in self.grid:
            c, s = math.cos(t / 2), math.sin(t / 2)
            p = c * c * a + s * s * b + 2 * c * s * r
            v = -float((cnt * np.log(np.maximum(p, 1e-12))).sum())
            if v < best_v:
                best_v, best_t = v, t
        from scipy.optimize import minimize_scalar

        def f(t):
            c, s = math.cos(t / 2), math.sin(t / 2)
            return -float((cnt * np.log(np.maximum(c * c * a + s * s * b + 2 * c * s * r, 1e-12))).sum())
        step = self.grid[1] - self.grid[0]
        res = minimize_scalar(f, bounds=(best_t - step, best_t + step), method='bounded',
                              options=dict(maxiter=25, xatol=1e-4))
        if res.fun < best_v:
            best_v, best_t = float(res.fun), float(res.x)
        return best_v, best_t

    def eval_insertions(self, arch, angles, data, picks):
        ops = sim8.fast_ops(arch, self.n)
        nz = np.nonzero(data.counts)
        cnt = data.counts[nz]
        Az = self._amps(ops, angles, data)[nz]
        a = Az.real ** 2 + Az.imag ** 2
        base = -float((cnt * np.log(np.maximum(a, 1e-12))).sum())
        out = []
        st = data.psi0.copy()
        cur = 0
        for p in sorted({p for p, _ in picks}):
            while cur < p:
                rows, ph = ops[cur]
                x = angles[cur]
                sim8._rot(st, rows, ph, math.cos(x / 2), math.sin(x / 2))
                cur += 1
            for pp, c in picks:
                if pp != p:
                    continue
                B = np.ascontiguousarray(self.ph_all[c][None, :] * st[:, self.rows_all[c]])
                for k in range(p, len(ops)):
                    rows, ph = ops[k]
                    x = angles[k]
                    sim8._rot(B, rows, ph, math.cos(x / 2), math.sin(x / 2))
                sim8._readout(B, data.mats, data.n)
                Bz = B[nz]
                b = Bz.real ** 2 + Bz.imag ** 2
                r = (Az * np.conj(-1j * Bz)).real
                v, t = self._curve_best(a, b, r, cnt)
                out.append((v - base, t, p, c))
        return out

    def _refit(self, arch, angles, data, maxiter):
        x, v = sim8.fast_fit(sim8.fast_ops(arch, self.n), angles, data, maxiter=maxiter)
        return x, v

    def _eliminate(self, arch, angles, data, cur, pen, deadline):
        """Delete gates whose removal costs less than the penalty (one at a time)."""
        for _ in range(3):
            if len(arch) < 2 or time.process_time() > deadline:
                break
            ops = sim8.fast_ops(arch, self.n)
            costs = []
            for k in range(len(arch)):
                x = angles.copy()
                x[k] = 0.0
                costs.append(sim8.fast_nll(ops, x, data) - cur)
            k = int(np.argmin(costs))
            if costs[k] >= pen:
                break
            arch = arch[:k] + arch[k + 1:]
            angles = np.delete(angles, k)
            angles, cur = self._refit(arch, angles, data, self.refit_iter)
            self.stats['deleted'] += 1
        return arch, angles, cur

    def run(self, data, deadline, arch=None, angles=None, trace=None):
        arch = list(arch or [])
        angles = np.asarray(angles if angles is not None else [], float)
        pen = self.penalty if self.penalty is not None else math.log(max(data.N, 2))
        cur = sim8.fast_nll(sim8.fast_ops(arch, self.n), angles, data) if arch else sim8.fast_nll([], [], data)
        while len(arch) < self.max_gates and time.process_time() < deadline:
            self.stats['steps'] += 1
            G = self.position_grads(arch, angles, data)
            ents = sum(len(t) == 2 for _, t in arch)
            if ents >= self.max_ents:
                for c, (nm, t) in enumerate(self.dict):
                    if len(t) == 2:
                        G[:, c] = 0
            flat = np.argsort(-np.abs(G).ravel())[: self.top]
            picks = [divmod(int(f), len(self.dict)) for f in flat]
            res = self.eval_insertions(arch, angles, data, picks)
            if not res:
                break
            dv, t, pos, cand = min(res)
            if -dv < pen:
                break
            arch = arch[:pos] + [self.dict[cand]] + arch[pos:]
            angles = np.concatenate([angles[:pos], [t], angles[pos:]])
            angles, cur = self._refit(arch, angles, data, self.refit_iter)
            self.stats['inserted'] += 1
            if self.prune and len(arch) >= 4:
                arch, angles, cur = self._eliminate(arch, angles, data, cur, pen, deadline)
            if trace is not None:
                trace.append((len(arch), round(cur, 1)))
        if arch:
            angles, cur = self._refit(arch, angles, data, 200)
        return arch, angles, cur
