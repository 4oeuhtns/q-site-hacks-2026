"""Open-final (8q) defender: f2def's frame learner plus a likelihood-selected pool.

Every candidate is a model of the ATTACK circuit; its patch is the exact inverse
(reversed gates, negated angles). Candidates are compared by BIC on all counts
collected so far, so a new source can only win by explaining the data better.

Sources:
  * f2def (Cup 2 frame learner, hardened) -- Clifford frames with few insertions.
  * Known public architectures (tlib.json) -- angles fitted by count likelihood,
    warm-started from each template's published range centres.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sim8  # noqa: E402
from duelkit.quantum import G, validate  # noqa: E402

# Template screening uses this many settings (the most-shot ones).
SCREEN_SETTINGS = 96
# Template angles may leave their published range by at most this much.
TEMPLATE_SLACK = 0.02
# A candidate with |gof z| below this explains the counts; skip costly sources.
GOOD_Z = 6.0
# Template / generator models enter the pool only below this gof z.
TEMPLATE_Z = 10.0
# Seed brute force of the public generators (checkpoint 1 only).
GEN_BUDGET = 60.0
GEN_SCREEN_SETTINGS = 8
# How many screened templates get a full angle fit.
FIT_TOP = 4
# CPU seconds for the template stage per checkpoint.
TEMPLATE_BUDGET = 45.0
# CPU seconds for unknown-architecture gate pursuit per checkpoint.
PURSUIT_BUDGET = {1: 150.0, 2: 130.0, 3: 110.0}
# Settings used by pursuit's inner loop (most-shot first); final refit uses all.
PURSUIT_SETTINGS = 160
# Whole-encounter CPU cap: optional sources are skipped beyond this.
ENCOUNTER_CAP = 480.0
LOG = []


def log(*a):
    LOG.append(' '.join(str(x) for x in a))
    print('opendef:', *a, file=sys.stderr)


class Model:
    def __init__(self, source, arch, angles, k=None):
        self.source = source
        self.arch = [(str(n), tuple(int(t) for t in ts)) for n, ts in arch]
        # R(t + 2 pi) = -R(t): wrapping is a global phase, and the rules reject
        # any patch angle outside [-pi, pi]. Unbounded refits drift past it.
        a = np.asarray(angles, float)
        self.angles = np.clip((a + math.pi) % (2 * math.pi) - math.pi, -math.pi, math.pi)
        self.k = len(self.angles) if k is None else k
        self.nll = math.inf

    def score(self, data):
        if not self.arch:
            self.nll = sim8.fast_nll([], [], data)
        else:
            self.nll = sim8.fast_nll(sim8.fast_ops(self.arch, data.n), self.angles, data)
        return self.nll

    def gof(self, data):
        try:
            return sim8.fast_gof(sim8.fast_ops(self.arch, data.n), self.angles, data)
        except Exception:
            return math.inf

    def bic(self, data):
        return 2 * self.nll + self.k * math.log(max(data.N, 2))

    def patch(self):
        return tuple(G(n, ts, float(-a)) for (n, ts), a in zip(reversed(self.arch), reversed(self.angles)))


def model_from_patch(source, patch):
    arch = [(g.name, g.targets) for g in reversed(patch)]
    ang = [-float(g.angle) for g in reversed(patch)]
    return Model(source, arch, ang)


class TemplateSource:
    def __init__(self, path=HERE / 'tlib.json'):
        try:
            self.lib = json.loads(Path(path).read_text())
        except Exception as exc:
            log('no template library', exc)
            self.lib = {}
        self.fitted = {}     # key -> angles
        self.shortlist = None

    def run(self, data, n, deadline):
        if not self.lib:
            return []
        order = np.argsort(-data.counts.sum(1))[:SCREEN_SETTINGS]
        sub = data.subset(order)
        scored = []
        keys = self.shortlist if self.shortlist is not None else list(self.lib)
        for key in keys:
            gates = self.lib[key]
            if time.process_time() > deadline:
                break
            arch = [(g[0], g[1]) for g in gates]
            x = self.fitted.get(key)
            if x is None:
                x = np.array([(g[2] + g[3]) / 2 for g in gates])
            try:
                v = sim8.fast_nll(sim8.fast_ops(arch, n), x, sub)
            except Exception:
                continue
            scored.append((v, key, x))
        scored.sort(key=lambda t: t[0])
        if self.shortlist is None:
            self.shortlist = [k for _, k, _ in scored[:FIT_TOP]]
        models = []
        for v, key, x in scored[:FIT_TOP]:
            if time.process_time() > deadline:
                break
            gates = self.lib[key]
            arch = [(g[0], g[1]) for g in gates]
            ops = sim8.fast_ops(arch, n)
            # A copied template draws every angle inside its published range; a
            # wrong architecture must not be allowed to bend into a free fit.
            bounds = [(g[2] - TEMPLATE_SLACK, g[3] + TEMPLATE_SLACK) for g in gates]
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                x1, _ = sim8.fast_fit(ops, x, sub, bounds=bounds, maxiter=200)
                x2, f = sim8.fast_fit(ops, x1, data, bounds=bounds, maxiter=200)
                z = sim8.fast_gof(ops, x2, data)
                if z > TEMPLATE_Z and time.process_time() < deadline:
                    # Right architecture with edited ranges: let the angles go free.
                    x3, f3 = sim8.fast_fit(ops, x2, data, maxiter=200)
                    z3 = sim8.fast_gof(ops, x3, data)
                    if z3 < z:
                        x2, f, z = x3, f3, z3
            except Exception as exc:
                log('fit failed', key, exc)
                continue
            self.fitted[key] = x2
            log(f'template {key}: screen {v:.1f} fitted {f:.1f} z={z:.1f}')
            # Only a template that explains the counts may compete: a wrong
            # architecture can still beat a failed pursuit on BIC.
            if z < TEMPLATE_Z:
                m = Model('template:' + key, arch, x2)
                m.nll = f
                models.append(m)
        return models


class GeneratorSource:
    """Brute-force the seeds of the SDK's public open-final generators.

    A team that calls open_example(rules, seed, ...) with its own seed publishes
    nothing we can list in advance, but the generator is public and cheap to
    replay. 8 settings rank the true seed first by a wide margin.
    """
    SIZES = ((36, 12), (72, 24), (18, 6))
    EXTRA_SIZES = ((48, 16), (24, 8), (60, 20), (12, 4), (54, 18))
    SPECIAL = (1234, 12345, 123456, 2024, 2025, 2026, 1337, 31337, 4242, 9999, 99999, 100000,
               314159, 271828, 8675309, 1000000, 65536, 4096, 2048, 1024, 777, 666, 888)

    def __init__(self, rules):
        from qduel_sdk.profiles import open_example, open_multilayer_example
        self.rules = rules
        self.ex, self.ml = open_example, open_multilayer_example
        self.best = None

    def keys(self):
        for s in range(128, 1000):
            for g, e in self.SIZES:
                yield ('ex', s, g, e)
            yield ('ml', s, 0, 0)
        for s in self.SPECIAL:
            for g, e in self.SIZES + self.EXTRA_SIZES:
                yield ('ex', s, g, e)
            yield ('ml', s, 0, 0)
        for s in range(0, 300):
            for g, e in self.EXTRA_SIZES:
                yield ('ex', s, g, e)

    def gates(self, key):
        kind, s, g, e = key
        t = self.ex(self.rules, s, gates=g, entanglers=e) if kind == 'ex' else self.ml(self.rules, s)
        return [(x['name'], x['targets'], x['low'], x['high']) for x in t['gates']]

    def run(self, data, deadline, stage):
        n = data.n
        if stage == 1:
            sub = data.subset(np.argsort(-data.counts.sum(1))[:GEN_SCREEN_SETTINGS])
            top = []
            count = 0
            for key in self.keys():
                if time.process_time() > deadline:
                    break
                try:
                    gs = self.gates(key)
                    arch = [(a, t) for a, t, _, _ in gs]
                    x = np.array([(lo + hi) / 2 for _, _, lo, hi in gs])
                    v = sim8.fast_nll(sim8.fast_ops(arch, n), x, sub)
                except Exception:
                    continue
                count += 1
                top.append((v, key, arch, x))
                top.sort(key=lambda t: t[0])
                del top[3:]
            log(f'generator screened {count} seeds; top {[(round(v), k) for v, k, _, _ in top]}')
            cands = top
        elif self.best is not None:
            cands = [(0.0,) + self.best]
        else:
            return []
        models = []
        for v, key, arch, x in cands:
            ops = sim8.fast_ops(arch, n)
            try:
                x2, f = sim8.fast_fit(ops, x, data, maxiter=200)
                z = sim8.fast_gof(ops, x2, data)
            except Exception as exc:
                log('generator fit failed', key, exc)
                continue
            log(f'generator {key}: nll {f:.1f} z={z:.1f}')
            if z < TEMPLATE_Z:
                m = Model(f'generator:{key}', arch, x2)
                m.nll = f
                models.append(m)
                self.best = (key, arch, x2)
                break
        return models


class PursuitSource:
    def __init__(self, n, rules):
        import pursuit
        kw = dict(max_gates=min(72, rules.patch_max_gates), max_ents=min(24, rules.patch_max_entanglers))
        # Two engines that win on different attacks (frontier benchmark):
        # p2 = closed-form insertion + deletion (fast), p1 = gradient-ranked line search.
        self.engines = {'p2': pursuit.Pursuit2(n, **kw), 'p1': pursuit.Pursuit(n, **kw)}
        self.schedule = {1: 'p2', 2: 'p1', 3: 'p2'}
        self.arch, self.angles = [], np.zeros(0)
        self.stage = 0

    def run(self, data, deadline):
        self.stage += 1
        out = []
        now = time.process_time()
        # A fresh fit on more data can escape an early wrong structure; the warm
        # continuation keeps what was already right. Both enter the pool.
        fresh_deadline = now + 0.65 * (deadline - now) if self.arch else deadline
        name = self.schedule.get(self.stage, 'p2')
        arch, ang, f = self.engines[name].run(data, fresh_deadline)
        if arch:
            m = Model(f'pursuit:{name}:fresh', arch, ang); m.nll = f; out.append(m)
            log(f'pursuit {name} fresh gates={len(arch)} nll={f:.1f}')
        if self.arch and time.process_time() < deadline:
            a2, x2, f2 = self.engines['p2'].run(data, deadline, self.arch, self.angles)
            m = Model('pursuit:p2:warm', a2, x2); m.nll = f2; out.append(m)
            log(f'pursuit warm gates={len(a2)} nll={f2:.1f}')
        if out:
            best = min(out, key=lambda m: m.nll)
            self.arch, self.angles = best.arch, best.angles
        return out


def legal(patch, rules):
    try:
        validate(patch, **rules.validation_kwargs())
        return True
    except Exception:
        return False


def run_defender(client, rules):
    import f2def

    class Bridge:
        def __init__(self):
            self.records = []; self.stage = 0; self.ceiling = 0; self.remaining = 0
            self.settings_remaining = 0; self.counter = 0; self.seen = set()

        def query(self, prep, basis, shots):
            from duelkit.quantum import experiment_index, bitstrings
            self.counter += 1
            basis = ''.join(basis)
            idx = experiment_index(prep, basis, rules.qubits)
            r = client.query(idx, int(shots), f'open-{self.stage}-{self.counter}')
            self.remaining -= int(shots)
            self.seen.add(idx)
            self.settings_remaining = rules.max_settings - len(self.seen)
            c = [r['counts'][b] for b in bitstrings(rules.qubits)]
            self.records.append(dict(prep=list(prep), basis=list(basis), counts=c, shots=int(shots), stage=self.stage))
            return np.array(c, dtype=np.int64)

    n = rules.qubits
    bridge = Bridge()
    learner = f2def.Frame2Recovery(n, rules.patch_max_gates, rules.patch_max_entanglers, seed=917)
    templates = TemplateSource()
    try:
        generator = GeneratorSource(rules)
    except Exception as exc:
        log('generator unavailable', exc)
        generator = None
    t_start = time.process_time()
    try:
        chaser = PursuitSource(n, rules)
    except Exception as exc:
        log('pursuit unavailable', exc)
        chaser = None
    last = ()
    for stage in range(1, rules.checkpoints + 1):
        state = client.status()
        bridge.stage = stage
        bridge.ceiling = stage * rules.block
        bridge.remaining = state['available_now']
        bridge.settings_remaining = rules.max_settings - state['distinct_settings']
        pool = []
        try:
            proposed, meta = learner.checkpoint(bridge)
            conv = []
            for g in proposed:
                active = [i for i, a in enumerate(g['pauli']) if a != 'I']
                axes = [g['pauli'][i].lower() for i in active]
                if not 1 <= len(active) <= 2 or len(set(axes)) != 1:
                    raise ValueError('illegal rotation')
                conv.append(G('r' + ''.join(axes), tuple(active), float(g['angle'])))
            pool.append(model_from_patch('f2def:' + str(meta.get('status')), tuple(conv)))
        except Exception as exc:
            log('f2def failed', type(exc).__name__, str(exc)[:200])
        # Spend anything the learner left (it normally spends the whole block).
        try:
            st = client.status()
            if st['available_now'] > 0 and bridge.records:
                keys = {}
                for r in bridge.records:
                    keys[(tuple(r['prep']), ''.join(r['basis']))] = r
                vals = list(keys.values())
                rem = int(st['available_now'])
                per = rem // len(vals)
                for i, r in enumerate(vals):
                    s = per + (1 if i < rem % len(vals) else 0)
                    if s:
                        bridge.query(r['prep'], r['basis'], s)
        except Exception as exc:
            log('top-up failed', exc)
        try:
            data = sim8.records_to_fast(bridge.records, n)
            if last:
                pool.append(model_from_patch('previous', last))
            pool.append(Model('identity', [], []))
            early = [m.gof(data) for m in pool if m.arch and math.isfinite(m.score(data))]
            explained = bool(early) and min(early) < GOOD_Z
            if explained:
                log(f'stage {stage} already explained (z={min(early):.1f}); optional sources skipped')
            else:
                pool.extend(templates.run(data, n, time.process_time() + TEMPLATE_BUDGET))
            for m in pool:
                if not math.isfinite(m.nll):
                    m.score(data)
            fits = [(m.gof(data), m) for m in pool if m.arch]
            best_z = min((z for z, _ in fits), default=math.inf)
            if generator is not None and best_z > GOOD_Z and time.process_time() - t_start < ENCOUNTER_CAP:
                try:
                    got = generator.run(data, time.process_time() + GEN_BUDGET, stage)
                    pool.extend(got)
                    fits += [(m.gof(data), m) for m in got]
                except Exception as exc:
                    log('generator failed', type(exc).__name__, str(exc)[:200])
            best_z = min((z for z, _ in fits), default=math.inf)
            log(f'stage {stage} best gof z before pursuit: {best_z:.1f}')
            used = time.process_time() - t_start
            if chaser is not None and used < ENCOUNTER_CAP and best_z > GOOD_Z:
                budget = min(PURSUIT_BUDGET.get(stage, 90.0), ENCOUNTER_CAP - used)
                try:
                    pool.extend(chaser.run(data, time.process_time() + budget))
                except Exception as exc:
                    log('pursuit failed', type(exc).__name__, str(exc)[:200])
            for m in pool:
                if not math.isfinite(m.nll) or m.source.startswith(('f2def', 'previous', 'identity')):
                    m.score(data)
            pool = [m for m in pool if math.isfinite(m.nll) and legal(m.patch(), rules)]
            pool.sort(key=lambda m: m.bic(data))
            for m in pool[:5]:
                log(f'stage {stage} {m.source:<40} nll={m.nll:.1f} bic={m.bic(data):.1f} k={m.k}')
            if pool:
                last = pool[0].patch()
        except Exception as exc:
            log('pool failed', type(exc).__name__, str(exc)[:200])
            if pool and legal(pool[0].patch(), rules):
                last = pool[0].patch()
        client.submit_patch(last, note='opendef')
        client.close_checkpoint()
