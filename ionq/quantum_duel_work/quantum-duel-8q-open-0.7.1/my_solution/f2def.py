"""Cup 2 (8q frame2) defender: the stock recovery8 learner with a hardened back end.

Built on the SDK's `Recovery` (parity discovery, residual algebra fit, atlas
factoring). Every change targets a way the stock learner loses points it could
have kept; each was found by a measured failure on a public or own frame case.

  * Fit basin. Stock warm-starts the residual fit from the previous
    checkpoint, which can pin a wrong basin (public_local@73: 15.6 pts at
    checkpoints 1-2). Here a fresh 40-start fit always races the warm start.
  * Hidden near-Clifford residual. When parity learning reaches full rank,
    stock submits a pure Clifford and loses any rotation within ~0.1 rad of a
    quarter turn (public_merged: 63-92 pts). Here the violations of the
    learned stabilizer map locate the residual Pauli, and its angle is fitted
    by count likelihood (`residual_rotation`).
  * Synthesis. Stock compiles its 16 candidates with up to 192 attempts each
    and no time bound (~1100 CPU s per encounter on compile-hard frames).
    Here the same 16 candidates are searched round-robin with fresh
    orderings, stopping at the first legal exact network or a CPU deadline.
  * Fallback. When nothing fits, stock resubmits the previous patch (empty at
    checkpoint 1). Here the smallest residual rotations are dropped until the
    network fits; losing a rotation by t costs sin^2(t/2).
  * Crashes. Any exception inside a checkpoint keeps the last legal patch.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from duelkit.quantum import G, validate, experiment_index, bitstrings
from duelkit.recovery8.recovery import Recovery, refine_factor
from duelkit.recovery8.quantum import unitary, inverse, canonical_angle
from duelkit.recovery8.quantum import validate as validate8
from duelkit.recovery8.clifford import parse, fmt, conjugate, images, complete_images, simplify
from duelkit.recovery8.algebra import Algebra, factor_paulis
from duelkit.recovery8.algebra_cliffords import clifford_atlas
from duelkit.recovery8.pauli_compile import row_image, clifford_absorb, merge_locals, qgate
from duelkit.recovery8.clifford_pair import pair_synthesis

SYNTH_BUDGET = 90.0
STAGE_BUDGET = {1: 150.0, 2: 100.0, 3: 60.0}
FALLBACK_BUDGET = 20.0
TAIL_TRIALS = (4, 8, 32)
BIC_MARGIN = 50.0
FRESH_STARTS = 40
FIT_CHUNK = 4
FIT_BUDGET = 45.0


def entanglers(gs):
    return sum(sum(c != 'I' for c in g['pauli']) == 2 for g in gs)


def cost_key(gs, gate_cap, ent_cap):
    e = entanglers(gs)
    return (int(e > ent_cap or len(gs) > gate_cap), e, len(gs))


def tail_synthesis(rows, n, rng, trials):
    """best_pair with a caller-supplied RNG, so repeated calls explore new orderings."""
    best = None
    for k in range(trials):
        order = list(rng.permutation(n))
        try:
            gs = pair_synthesis(rows, n, order)
        except (KeyError, AssertionError, StopIteration):
            continue
        key = (entanglers(gs), len(gs))
        if best is None or key < best[0]:
            best = (key, gs)
    if best is None:
        raise ValueError('paired synthesis failed')
    return best[1]


def network(paulis, tail_rows, n, rng, tail_trials):
    """pauli_compile.network with a live RNG for both pivots and tail orderings."""
    F = []
    prefix = []
    for orig in paulis:
        p = row_image(F, (*parse(orig['pauli']), 1))
        s = fmt(p[0], p[1], n)
        supp = [i for i, c in enumerate(s) if c != 'I']
        if not supp:
            continue
        pivot = int(rng.choice(supp))
        leaves = [q for q in supp if q != pivot]
        rng.shuffle(leaves)
        for q in leaves:
            s = fmt(p[0], p[1], n)
            a = s[q]
            if s[pivot] == a:
                change = str(rng.choice([x for x in 'XYZ' if x != a]))
                g = qgate(n, pivot, change, np.pi / 2)
                F.append(g); prefix.append(g); p = conjugate(g, p)
            g = {'pauli': ''.join(a if i in (pivot, q) else 'I' for i in range(n)), 'angle': np.pi / 2}
            F.append(g); prefix.append(g); p = conjugate(g, p)
        s = fmt(p[0], p[1], n)
        prefix.append({'pauli': s, 'angle': canonical_angle(float(p[2]) * orig['angle'])})
    standard = images([], n)
    t_inverse = complete_images(list(zip(tail_rows, standard)), n)
    target_inv = [row_image(F, p) for p in t_inverse]
    gs = tail_synthesis(target_inv, n, rng, tail_trials)
    return merge_locals(simplify(prefix + gs, n), n)


def option_key(non, rows):
    """Network structure depends on the rotation Paulis and the tail, not on the angles."""
    return (tuple(g['pauli'] for g in non), tuple(rows))


def search(options, n, gate_cap, ent_cap, deadline, rng, cache=None):
    """Cheapest network over a list of (non, rows, lost) options until the deadline.

    Each option is tried at least once, in order, before any is retried.
    `lost` is the infidelity the option accepts by dropping rotations; legal
    networks with smaller loss always beat cheaper ones with larger loss.
    """
    best = None
    rounds = 0
    cache = {} if cache is None else cache
    for non, rows, lost in options:
        hit = cache.get(option_key(non, rows))
        if hit is None:
            continue
        try:
            gs = network(non, rows, n, np.random.default_rng(hit[0]), tail_trials=hit[1])
        except (ValueError, AssertionError, StopIteration, KeyError):
            continue
        key = cost_key(gs, gate_cap, ent_cap)
        score = (key[0], round(lost, 12), key[1], key[2])
        if best is None or score < best[0]:
            best = (score, gs, lost)
    if best and best[0][0] == 0 and best[2] == 0:
        return best, 0
    while True:
        tail_trials = TAIL_TRIALS[rounds % len(TAIL_TRIALS)]
        for non, rows, lost in options:
            seed = int(rng.integers(1 << 62))
            try:
                gs = network(non, rows, n, np.random.default_rng(seed), tail_trials=tail_trials)
            except (ValueError, AssertionError, StopIteration, KeyError):
                continue
            if not cost_key(gs, gate_cap, ent_cap)[0]:
                cache[option_key(non, rows)] = (seed, tail_trials)
            key = cost_key(gs, gate_cap, ent_cap)
            score = (key[0], round(lost, 12), key[1], key[2])
            if best is None or score < best[0]:
                best = (score, gs, lost)
            if best[0][0] == 0 and best[2] == 0:
                return best, rounds + 1
            if time.process_time() > deadline:
                break
        rounds += 1
        if time.process_time() > deadline:
            break
        if best and best[0][0] == 0 and not any(o[2] == 0 for o in options):
            break
    return best, rounds


def dropped_options(non, rows):
    """Options that drop the k smallest residual rotations, k = 0..len(non)."""
    order = sorted(range(len(non)), key=lambda i: abs(non[i]['angle']))
    out = []
    for k in range(len(non) + 1):
        gone = set(order[:k])
        kept = [g for i, g in enumerate(non) if i not in gone]
        lost = 1 - float(np.prod([np.cos(non[i]['angle'] / 2) ** 2 for i in gone])) if gone else 0.0
        out.append((kept, rows, lost))
    return out


class Frame2Recovery(Recovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_cache = {}

    def checkpoint(self, client):
        try:
            return self._checkpoint(client)
        except Exception as exc:
            return self.patch, {'status': 'EXCEPTION_KEPT_LAST', 'error': f'{type(exc).__name__}: {exc}'[:200]}

    def _submit_candidates(self, options, meta, start):
        rng = np.random.default_rng(self.seed + 7919 * getattr(self, '_stage', 0))
        deadline = time.process_time() + STAGE_BUDGET.get(getattr(self, '_stage', 0), SYNTH_BUDGET)
        lossless = [o for o in options if o[2] == 0]
        best, rounds = search(lossless, self.n, self.gc, self.ec, deadline, rng, self.net_cache)
        if best is None or best[0][0]:
            lossy = [o for o in options if o[2] > 0]
            fallback, more = search(lossy, self.n, self.gc, self.ec, time.process_time() + FALLBACK_BUDGET, rng, self.net_cache)
            rounds += more
            if fallback is not None and (best is None or fallback[0] < best[0]):
                best = fallback
        meta['synth_rounds'] = rounds
        if best is None or best[0][0]:
            meta['status'] = 'ALL_OVER_BUDGET_KEPT_LAST'
            return
        gs = best[1]
        validate8(gs, self.n, self.gc, self.ec)
        self.patch = gs
        meta.update(status='LEGAL_COMPILED' if best[2] == 0 else 'LEGAL_DROPPED_ROTATIONS',
                    accepted_loss=best[2], entanglers=best[0][2], gates=best[0][3])

    def _checkpoint(self, client):
        self._stage = client.stage
        start = time.process_time()
        self.parity.run_until(client, client.ceiling)
        if client.remaining:
            history = client.records
            distinct = {(tuple(r['prep']), tuple(r['basis'])): r for r in history}
            vals = list(distinct.values())
            remaining = client.remaining
            if vals:
                alloc = np.full(len(vals), remaining // len(vals), int)
                alloc[:remaining % len(vals)] += 1
                for r, s in zip(vals, alloc):
                    if s:
                        client.query(r['prep'], r['basis'], int(s))
        n = self.n
        rank = self.parity.rank
        meta = {'rank': rank, 'n_learned_pairs': len(self.parity.pairs)}
        if rank == 2 * n:
            rows = complete_images(self.parity.pairs, n)
            options = [([], pure_tail_rows(rows, n), 0.0)]
            try:
                found = residual_rotation(rows, client.records, n, np.random.default_rng(self.seed))
            except Exception as exc:
                found = None
                meta['residual_error'] = f'{type(exc).__name__}: {exc}'[:200]
            if found is not None:
                patch_c, label, delta, gain = found
                non, rows2 = clifford_absorb(patch_c + [{'pauli': label, 'angle': -delta}], [], n, 1e-8)
                options = [(non, rows2, 0.0)] + [(o[0], o[1], 1.0) for o in options]
                meta.update(residual_pauli=label, residual_angle=delta, residual_loglik_gain=gain)
            self._submit_candidates(options, meta, start)
            return self.patch, meta
        if rank < 2 * n - 4:
            meta['status'] = 'RESIDUAL_ALGEBRA_TOO_LARGE'
            return self.patch, meta
        signature = str(self.parity.pairs)
        if signature != self.oldpairs:
            self.model = Algebra(n, self.parity.pairs)
            self.oldpairs = signature
            self.oldx = None
            self.atlas, self.words = clifford_atlas(self.model.L)
        m = self.model
        data = {}
        for r in client.records:
            key = (tuple(r['prep']), tuple(r['basis']))
            if key not in data:
                data[key] = np.zeros(1 << n, dtype=np.int64)
            data[key] += np.array(r['counts'])
        settings = [{'prep': list(a), 'basis': list(b)} for a, b in data]
        counts = np.stack(list(data.values()))
        fits = []
        if self.oldx is not None:
            fits.append(m.fit(settings, counts, seed=self.seed + client.stage, starts=5, xold=self.oldx))
        fit_deadline = time.process_time() + FIT_BUDGET
        for chunk in range(0, FRESH_STARTS, FIT_CHUNK):
            fits.append(m.fit(settings, counts, seed=self.seed + 101 * client.stage + chunk,
                              starts=min(FIT_CHUNK, FRESH_STARTS - chunk)))
            if time.process_time() > fit_deadline:
                break
        V, fit = min(fits, key=lambda vf: vf[1]['loss'])
        self.oldx = np.array(fit['x'])
        T = m.features(settings)
        ct = counts.ravel().astype(float)
        N = ct.sum()
        top = np.argsort(abs(self.atlas.conj() @ V[:, 0]) ** 2)[::-1][:12]
        candidates = []
        seen = set()
        for ai in top:
            D = np.einsum('j,jab->ab', self.atlas[ai], m.L)
            W = D.conj().T @ V
            ids, x, approx = factor_paulis(W.conj().T, m.L, maxg=10, tol=max(4 / N, 3e-5))
            dw = [{'pauli': m.labels[z], 'angle': np.pi / 2} for z in self.words[ai]]
            tail = m.cinv + inverse(dw)
            for threshold in [.025, .05]:
                xx, loss, bic, k = refine_factor(m, D, ids, x, threshold, T, ct)
                high = []
                for i, ang in zip(ids, xx):
                    ang = canonical_angle(ang)
                    if abs(ang) < 1e-10:
                        continue
                    row = row_image(inverse(tail), (*parse(m.labels[i]), 1))
                    high.append({'pauli': fmt(row[0], row[1], n),
                                 'angle': canonical_angle(float(ang) * float(row[2]))})
                key = str(high) + str(tail)
                if key not in seen:
                    candidates.append((bic, loss, k, high, tail))
                    seen.add(key)
        candidates.sort(key=lambda t: t[0])
        if not candidates:
            meta['status'] = 'NO_CANDIDATES_KEPT_LAST'
            return self.patch, meta
        best_bic = candidates[0][0]
        near = [c for c in candidates if c[0] <= best_bic + BIC_MARGIN][:16]
        self.last_candidates = candidates
        self.last_fit = fit
        options = []
        for bic, loss, k, high, tail in near:
            non, rows = clifford_absorb(high, tail, n, 1e-8)
            options.extend(dropped_options(non, rows))
        meta['near_candidates'] = len(near)
        self._submit_candidates(options, meta, start)
        meta['cpu_seconds'] = time.process_time() - start
        return self.patch, meta


STAB = {'0': ('Z', 1), '1': ('Z', -1), '+': ('X', 1), '-': ('X', -1), '+i': ('Y', 1), '-i': ('Y', -1)}
RESIDUAL_MIN_GAIN = 20.0


def _single(letter, q, n):
    bit = 1 << (n - q - 1)
    return (bit if letter in 'XY' else 0, bit if letter in 'YZ' else 0)


def stabilizer_items(rows, records, n):
    """(Sx, Sz, shots, violations) for every prep-stabilizer whose image is diagonal in the readout.

    Under A = C W with W = exp(-i d Q / 2), a stabilizer S of the product input
    keeps expectation +/-1 through C unless S anticommutes with Q, in which case
    it drops to cos d. So in noiseless counts a violation marks S as
    anticommuting with Q, and the violation rate is sin^2(d/2).
    """
    from duelkit.recovery8.clifford import pmul
    img = {}
    for q in range(n):
        ix, iz = rows[q], rows[n + q]
        img[q, 'X'] = ix
        img[q, 'Z'] = iz
        y = pmul(ix, iz)
        img[q, 'Y'] = (y[0], y[1], 1j * y[2])
    data = {}
    for r in records:
        key = (tuple(r['prep']), tuple(r['basis']))
        data.setdefault(key, np.zeros(1 << n, dtype=np.int64))
        data[key] += np.asarray(r['counts'])
    outcomes = np.arange(1 << n)
    items = []
    for (prep, basis), counts in data.items():
        shots = int(counts.sum())
        stabs = [STAB[p] for p in prep]
        for mask in range(1, 1 << n):
            sx = sz = 0
            acc = (0, 0, 1)
            sign = 1
            for q in range(n):
                if mask >> (n - q - 1) & 1:
                    letter, s = stabs[q]
                    x, z = _single(letter, q, n)
                    sx |= x; sz |= z; sign *= s
                    acc = pmul(acc, img[q, letter])
            ox, oz, ph = acc
            ph = complex(ph) * sign
            if abs(ph.imag) > 1e-8:
                continue
            diagonal = True
            supp = 0
            for q in range(n):
                bit = 1 << (n - q - 1)
                a = 'Y' if (ox & oz & bit) else 'X' if (ox & bit) else 'Z' if (oz & bit) else 'I'
                if a == 'I':
                    continue
                if a != basis[q]:
                    diagonal = False
                    break
                supp |= bit
            if not diagonal or not supp:
                continue
            parity = np.array([bin(o & supp).count('1') & 1 for o in outcomes])
            expected_bit = 0 if ph.real > 0 else 1
            viol = int(counts[parity != expected_bit].sum())
            items.append((sx, sz, shots, viol))
    return items


def residual_rotation(rows, records, n, rng):
    """Find W = exp(-i d Q/2) with A = C W from stabilizer violations, then fit d by likelihood.

    Returns (clifford_patch, Q, d, loglik_gain) or None when no residual is supported.
    """
    from duelkit.recovery8.quantum import probabilities
    items = stabilizer_items(rows, records, n)
    if not items:
        return None
    arr = np.array(items, dtype=np.int64)
    sx, sz, shots, viol = arr.T
    if viol.sum() == 0:
        return None
    qs = np.arange(1, 1 << (2 * n), dtype=np.int64)
    qx, qz = qs & ((1 << n) - 1), qs >> n
    best = None
    for lo in range(0, len(qs), 4096):
        bx, bz = qx[lo:lo + 4096, None], qz[lo:lo + 4096, None]
        w = (sx[None, :] & bz) ^ (sz[None, :] & bx)
        anti = np.zeros(w.shape, dtype=bool)
        for b in range(n):
            anti ^= (w >> b & 1).astype(bool)
        feasible = ~((viol[None, :] > 0) & ~anti).any(axis=1)
        if not feasible.any():
            continue
        Na = (anti * shots[None, :]).sum(axis=1)
        Va = (anti * viol[None, :]).sum(axis=1)
        p = np.clip(Va / np.maximum(Na, 1), 1e-9, 1 - 1e-9)
        ll = np.where(feasible, Va * np.log(p) + (Na - Va) * np.log(1 - p), -np.inf)
        i = int(np.argmax(ll))
        if best is None or ll[i] > best[0]:
            best = (float(ll[i]), int(qx[lo + i]), int(qz[lo + i]), float(p[i]))
    if best is None:
        return None
    _, x, z, p = best
    label = fmt(x, z, n)
    patch_c = tail_synthesis(rows, n, rng, 16)
    settings, counts = [], []
    data = {}
    for r in records:
        key = (tuple(r['prep']), tuple(r['basis']))
        data.setdefault(key, np.zeros(1 << n, dtype=np.int64))
        data[key] += np.asarray(r['counts'])
    settings = [{'prep': list(a), 'basis': list(b)} for a, b in data]
    counts = np.stack(list(data.values())).astype(float)
    clifford = inverse(patch_c)

    def loglik(d):
        probs = probabilities([{'pauli': label, 'angle': float(d)}] + clifford, n, settings)
        return float((counts * np.log(probs + 1e-12)).sum())

    base = loglik(0.0)
    mag = 2 * np.arcsin(np.sqrt(p))
    d0 = max((mag, -mag), key=loglik)
    a, b = d0 - 0.5 * abs(d0) - 0.02, d0 + 0.5 * abs(d0) + 0.02
    g = (np.sqrt(5) - 1) / 2
    c, e = b - g * (b - a), a + g * (b - a)
    fc, fe = loglik(c), loglik(e)
    for _ in range(18):
        if fc > fe:
            b, e, fe = e, c, fc
            c = b - g * (b - a); fc = loglik(c)
        else:
            a, c, fc = c, e, fe
            e = a + g * (b - a); fe = loglik(e)
    d = c if fc > fe else e
    gain = max(fc, fe) - base
    if gain < RESIDUAL_MIN_GAIN:
        return None
    return patch_c, label, float(d), float(gain)


def pure_tail_rows(rows, n):
    """Forward tail rows for the inverse of a learned pure Clifford.

    `complete_images` gives the attack's forward images; the stock learner
    synthesises its inverse with best_pair(rows). network() expects the forward
    action of the desired trailing Clifford, whose inverse is then synthesised,
    so the tail must be the attack inverse, i.e. the inverse tableau of rows.
    """
    standard = images([], n)
    return complete_images(list(zip(rows, standard)), n)


def run_defender(client, rules):
    if rules.qubits != 8:
        raise ValueError('Frame defender expects the eight-qubit profile')

    class Bridge:
        def __init__(self):
            self.records = []; self.stage = 0; self.ceiling = 0; self.remaining = 0
            self.settings_remaining = 0; self.counter = 0; self.seen = set()

        def query(self, prep, basis, shots):
            self.counter += 1
            basis = ''.join(basis)
            idx = experiment_index(prep, basis, rules.qubits)
            r = client.query(idx, int(shots), f'frame-{self.stage}-{self.counter}')
            self.remaining -= int(shots)
            self.seen.add(idx)
            self.settings_remaining = rules.max_settings - len(self.seen)
            c = [r['counts'][b] for b in bitstrings(rules.qubits)]
            self.records.append(dict(prep=list(prep), basis=list(basis), counts=c, shots=int(shots), stage=self.stage))
            return np.array(c, dtype=np.int64)

    bridge = Bridge()
    learner = Frame2Recovery(8, rules.patch_max_gates, rules.patch_max_entanglers, seed=917)
    last = ()
    for stage in range(1, rules.checkpoints + 1):
        state = client.status()
        bridge.stage = stage
        bridge.ceiling = stage * rules.block
        bridge.remaining = state['available_now']
        bridge.settings_remaining = rules.max_settings - state['distinct_settings']
        proposed, meta = learner.checkpoint(bridge)
        try:
            converted = []
            for g in proposed:
                active = [i for i, a in enumerate(g['pauli']) if a != 'I']
                axes = [g['pauli'][i].lower() for i in active]
                if not 1 <= len(active) <= 2 or len(set(axes)) != 1:
                    raise ValueError('Compiler returned an illegal physical rotation')
                converted.append(G('r' + ''.join(axes), tuple(active), float(g['angle'])))
            converted = tuple(converted)
            validate(converted, **rules.validation_kwargs())
            last = converted
        except Exception:
            pass
        client.submit_patch(last, note='frame2 hardened recovery; ' + str(meta.get('status', 'checkpoint')))
        client.close_checkpoint()
