"""Batched n-qubit rotation-circuit simulator with an adjoint count-likelihood gradient.

Conventions match duelkit.quantum: q0 is the most significant bit, a rotation is
R_P(t) = cos(t/2) I - i sin(t/2) P, and P|v>[i] = phase[i] * v[rows[i]].
States are (2**n, S): one column per measured setting.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

_S2 = 1 / math.sqrt(2)
KET = {
    'Z+': np.array([1, 0], complex), 'Z-': np.array([0, 1], complex),
    'X+': np.array([1, 1], complex) * _S2, 'X-': np.array([1, -1], complex) * _S2,
    'Y+': np.array([1, 1j], complex) * _S2, 'Y-': np.array([1, -1j], complex) * _S2,
}
ALIAS = {'0': 'Z+', '1': 'Z-', '+': 'X+', '-': 'X-', '+i': 'Y+', '-i': 'Y-'}
READ = {b: np.stack([KET[b + '+'], KET[b + '-']]).conj() for b in 'XYZ'}


@lru_cache(maxsize=4096)
def pauli_rows(axis, targets, n):
    xmask = sum(1 << (n - 1 - q) for q in targets) if axis in 'xy' else 0
    ids = np.arange(1 << n)
    rows = ids ^ xmask
    phase = np.ones(1 << n, dtype=complex)
    for q in targets:
        bit = (rows >> (n - 1 - q)) & 1
        if axis == 'y':
            phase *= 1j * (1 - 2 * bit)
        elif axis == 'z':
            phase *= 1 - 2 * bit
    rows.flags.writeable = False
    phase.flags.writeable = False
    return rows, phase[:, None]


def arch_ops(arch, n):
    """arch: sequence of (name, targets). Returns list of (rows, phase)."""
    return [pauli_rows(name[-1], tuple(int(t) for t in targets), n) for name, targets in arch]


class Data:
    """Aggregated counts per distinct (prep, basis) setting, with cached prep/readout."""

    def __init__(self, preps, bases, counts, n):
        self.n = n
        self.counts = np.asarray(counts, dtype=float)
        self.N = self.counts.sum()
        S = len(preps)
        psi = np.ones((1, S), complex)
        for q in range(n):
            v = np.stack([KET[ALIAS.get(p[q], p[q])] for p in preps], axis=1)
            psi = (psi[:, None, :] * v[None, :, :]).reshape(-1, S)
        self.psi0 = np.ascontiguousarray(psi)
        self.read = [np.stack([READ[b[q]] for b in bases]) for q in range(n)]
        nz = self.counts > 0
        self.const = float((self.counts[nz] * np.log(self.counts[nz] / self.counts.sum(1, keepdims=True).repeat(1 << n, 1)[nz])).sum())

    def readout(self, state, adjoint=False):
        n, S = self.n, state.shape[1]
        t = state
        for q in range(n):
            m = self.read[q]
            if adjoint:
                m = np.conj(np.swapaxes(m, 1, 2))
            t = t.reshape(1 << q, 2, 1 << (n - q - 1), S)
            t = np.einsum('sab,ibjs->iajs', m, t)
        return t.reshape(1 << n, S)


def forward(ops, angles, psi):
    s = psi
    for (rows, ph), a in zip(ops, angles):
        s = math.cos(a / 2) * s - 1j * math.sin(a / 2) * (ph * s[rows])
    return s


def probs(ops, angles, data):
    phi = data.readout(forward(ops, angles, data.psi0))
    p = (phi.real ** 2 + phi.imag ** 2).T
    return p


def nll(ops, angles, data, grad=False, floor=1e-12):
    """Negative log-likelihood of the counts (minus the saturated constant)."""
    psiK = forward(ops, angles, data.psi0)
    phi = data.readout(psiK)
    p = phi.real ** 2 + phi.imag ** 2
    c = data.counts.T
    pc = np.maximum(p, floor)
    value = -float((c * np.log(pc)).sum()) + data.const
    if not grad:
        return value
    g = -(c / pc) * phi
    lam = data.readout(g, adjoint=True)
    psi = psiK
    out = np.zeros(len(ops))
    for k in range(len(ops) - 1, -1, -1):
        rows, ph = ops[k]
        a = angles[k]
        ca, sa = math.cos(a / 2), math.sin(a / 2)
        psi = ca * psi + 1j * sa * (ph * psi[rows])
        Ppsi = ph * psi[rows]
        d = -0.5 * sa * psi - 0.5j * ca * Ppsi
        out[k] = 2 * float(np.real(np.vdot(lam, d)))
        lam = ca * lam + 1j * sa * (ph * lam[rows])
    return value, out


def fit(ops, x0, data, bounds=None, maxiter=300):
    from scipy.optimize import minimize
    r = minimize(lambda x: nll(ops, x, data, grad=True), np.asarray(x0, float), jac=True,
                 method='L-BFGS-B', bounds=bounds, options=dict(maxiter=maxiter))
    return r.x, float(r.fun)


def records_to_data(records, n):
    agg = {}
    for r in records:
        key = (tuple(r['prep']), ''.join(r['basis']))
        c = np.asarray(r['counts'], dtype=float)
        agg[key] = agg.get(key, 0) + c
    keys = list(agg)
    return Data([k[0] for k in keys], [k[1] for k in keys], np.stack([agg[k] for k in keys]), n)


try:
    import numba as _nb

    @_nb.njit(cache=False)
    def _rot(st, rows, ph, c, s):
        S, D = st.shape
        for j in range(S):
            for i in range(D):
                r = rows[i]
                if r == i:
                    st[j, i] = (c - 1j * s * ph[i]) * st[j, i]
                elif i < r:
                    a = st[j, i]
                    b = st[j, r]
                    st[j, i] = c * a - 1j * s * ph[i] * b
                    st[j, r] = c * b - 1j * s * ph[r] * a

    @_nb.njit(cache=False)
    def _readout(st, mats, n):
        S, D = st.shape
        for j in range(S):
            for q in range(n):
                bit = 1 << (n - 1 - q)
                m00 = mats[j, q, 0, 0]; m01 = mats[j, q, 0, 1]
                m10 = mats[j, q, 1, 0]; m11 = mats[j, q, 1, 1]
                for i in range(D):
                    if i & bit == 0:
                        a = st[j, i]
                        b = st[j, i | bit]
                        st[j, i] = m00 * a + m01 * b
                        st[j, i | bit] = m10 * a + m11 * b

    @_nb.njit(cache=False)
    def _grad_term(lam, psi, rows, ph, c, s):
        S, D = psi.shape
        acc = 0.0
        for j in range(S):
            for i in range(D):
                d = -0.5 * s * psi[j, i] - 0.5j * c * ph[i] * psi[j, rows[i]]
                v = lam[j, i].conjugate() * d
                acc += v.real
        return 2.0 * acc

    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


class FastData:
    """Same as Data but (S, D) layout and numba kernels."""

    def __init__(self, preps, bases, counts, n):
        self.n = n
        self.counts = np.ascontiguousarray(np.asarray(counts, dtype=float))
        self.N = self.counts.sum()
        S = len(preps)
        psi = np.ones((S, 1), complex)
        for q in range(n):
            v = np.stack([KET[ALIAS.get(p[q], p[q])] for p in preps])
            psi = (psi[:, :, None] * v[:, None, :]).reshape(S, -1)
        self.psi0 = np.ascontiguousarray(psi)
        self.mats = np.ascontiguousarray(np.stack([[READ[b[q]] for q in range(n)] for b in bases]))
        self.mats_adj = np.ascontiguousarray(np.conj(np.swapaxes(self.mats, 2, 3)))
        rs = self.counts.sum(1, keepdims=True)
        nz = self.counts > 0
        self.const = float((self.counts[nz] * np.log((self.counts / np.maximum(rs, 1))[nz])).sum())

    def subset(self, idx):
        new = object.__new__(FastData)
        new.n = self.n
        new.counts = np.ascontiguousarray(self.counts[idx])
        new.N = new.counts.sum()
        new.psi0 = np.ascontiguousarray(self.psi0[idx])
        new.mats = np.ascontiguousarray(self.mats[idx])
        new.mats_adj = np.ascontiguousarray(self.mats_adj[idx])
        rs = new.counts.sum(1, keepdims=True)
        nz = new.counts > 0
        new.const = float((new.counts[nz] * np.log((new.counts / np.maximum(rs, 1))[nz])).sum())
        return new


def fast_ops(arch, n):
    out = []
    for name, targets in arch:
        rows, ph = pauli_rows(name[-1], tuple(int(t) for t in targets), n)
        out.append((np.ascontiguousarray(rows.astype(np.int64)), np.ascontiguousarray(ph[:, 0])))
    return out


def fast_state(ops, angles, data):
    st = data.psi0.copy()
    for (rows, ph), a in zip(ops, angles):
        _rot(st, rows, ph, math.cos(a / 2), math.sin(a / 2))
    return st


def fast_nll(ops, angles, data, grad=False, floor=1e-12):
    st = fast_state(ops, angles, data)
    phi = st.copy()
    _readout(phi, data.mats, data.n)
    p = phi.real ** 2 + phi.imag ** 2
    pc = np.maximum(p, floor)
    value = -float((data.counts * np.log(pc)).sum()) + data.const
    if not grad:
        return value
    lam = np.ascontiguousarray(-(data.counts / pc) * phi)
    _readout(lam, data.mats_adj, data.n)
    out = np.zeros(len(ops))
    for k in range(len(ops) - 1, -1, -1):
        rows, ph = ops[k]
        a = angles[k]
        c, s = math.cos(a / 2), math.sin(a / 2)
        _rot(st, rows, ph, c, -s)
        out[k] = _grad_term(lam, st, rows, ph, c, s)
        _rot(lam, rows, ph, c, -s)
    return value, out


def fast_fit(ops, x0, data, bounds=None, maxiter=300):
    from scipy.optimize import minimize
    r = minimize(lambda x: fast_nll(ops, x, data, grad=True), np.asarray(x0, float), jac=True,
                 method='L-BFGS-B', bounds=bounds, options=dict(maxiter=maxiter))
    return r.x, float(r.fun)


def records_to_fast(records, n):
    agg = {}
    for r in records:
        key = (tuple(r['prep']), ''.join(r['basis']))
        c = np.asarray(r['counts'], dtype=float)
        agg[key] = agg.get(key, 0) + c
    keys = list(agg)
    return FastData([k[0] for k in keys], [k[1] for k in keys], np.stack([agg[k] for k in keys]), n)


def fast_gof(ops, angles, data, floor=1e-12):
    """z-score of the observed log-likelihood against its expectation under the model.

    E[-sum n log p] = sum_s N_s H(p_s); Var = sum_s N_s Var_{p_s}(log p). |z| of a few
    means the model explains the counts to within shot noise.
    """
    st = fast_state(ops, angles, data)
    sim8_read = st
    _readout(sim8_read, data.mats, data.n)
    p = np.maximum(st.real ** 2 + st.imag ** 2, floor)
    p = p / p.sum(1, keepdims=True)
    lp = np.log(p)
    Ns = data.counts.sum(1)
    obs = -float((data.counts * lp).sum())
    H = -(p * lp).sum(1)
    V = (p * lp ** 2).sum(1) - H ** 2
    exp = float((Ns * H).sum())
    sd = math.sqrt(max(float((Ns * V).sum()), 1e-9))
    return (obs - exp) / sd
