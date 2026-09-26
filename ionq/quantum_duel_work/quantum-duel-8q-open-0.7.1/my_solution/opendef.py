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
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import sim8
import subsys
from duelkit.quantum import G, validate

SCREEN_SETTINGS = 96
TEMPLATE_SLACK = 0.02
GOOD_Z = 3.0
REFIT_TOP = 2
REFIT_ITER = 60
SEP_RATIO = 0.8
F2_STAGE1_SETTINGS = 180
F2_STAGE1_SHOTS = 16000
RANDOM_MIN_SHOTS = 40
FRAME_RANK = 8
EXACT_SOURCES = ('f2def:LEGAL_COMPILED', 'template:', 'generator:', 'subsys:', 'refit:template', 'refit:generator', 'refit:subsys')
SUBSYS_MAX = 3
SUBSYS_BUDGET = 40.0
TEMPLATE_Z = 10.0
GEN_BUDGET = 60.0
GEN_SCREEN_SETTINGS = 8
FIT_TOP = 4
TEMPLATE_BUDGET = 45.0
PURSUIT_BUDGET = {1: 110.0, 2: 90.0, 3: 90.0}
HARD_WARM_BUDGET = 30.0
WARM_CHECK_BUDGET = 60.0
REFIT_RESERVE = 40.0
NEAR_NAMED = ('bank:mixed', 'bank:multilayer', 'notebook:', 'open_example:23:', 'open_example:31:',
              'multilayer:37')
NEAR_SEP = 0.9
SEED_MIN_GATES = 12
SEED_BUDGET = 90.0
VARIANT_KEYS = ('bank:mixed', 'bank:multilayer', 'notebook:open_demo',
                'open_example:23:36:12', 'open_example:23:72:24', 'open_example:23:18:6')
F2_BUDGETS = {1: (150.0, 45.0, 20.0), 2: (100.0, 45.0, 20.0), 3: (60.0, 45.0, 20.0)}
F2_RESERVE = {2: 120.0, 3: 30.0}
F2_MIN = 30.0
PURSUIT_SETTINGS = 160
ENCOUNTER_CAP = 540.0
VERBOSE = bool(os.environ.get('OPENDEF_VERBOSE'))


def log(*a):
    if VERBOSE:
        print('opendef:', *a, file=sys.stderr)


class Model:
    def __init__(self, source, arch, angles, k=None):
        self.source = source
        self.arch = [(str(n), tuple(int(t) for t in ts)) for n, ts in arch]
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


def free_mask(angles):
    """True where an angle is not a multiple of pi/2 (Clifford angles are exact, not free)."""
    q = np.asarray(angles, float) / (math.pi / 2)
    return np.abs(q - np.round(q)) > 1e-7


def model_from_patch(source, patch):
    arch = [(g.name, g.targets) for g in reversed(patch)]
    ang = [-float(g.angle) for g in reversed(patch)]
    m = Model(source, arch, ang)
    m.k = int(free_mask(m.angles).sum())
    return m


def separation(scored):
    """scored: sorted [(nll, key, gates)] -> best / next-different-architecture ratio."""
    if len(scored) < 2:
        return 0.0 if scored else math.inf
    sig = lambda gates: tuple((g[0], tuple(g[1])) for g in gates)
    best = sig(scored[0][2])
    for v, _, gates in scored[1:]:
        if sig(gates) != best:
            return scored[0][0] / max(v, 1e-9)
    return 0.0


class TemplateSource:
    def __init__(self, path=HERE / 'tlib.json'):
        try:
            self.lib = json.loads(Path(path).read_text())
        except Exception as exc:
            log('no template library', exc)
            self.lib = {}
        for key in VARIANT_KEYS:
            gates = self.lib.get(key)
            if gates:
                neg = [[g[0], g[1], -g[3], -g[2]] for g in gates]
                self.lib['neg:' + key] = neg
                self.lib['rev:' + key] = [list(g) for g in reversed(gates)]
                self.lib['inv:' + key] = neg[::-1]
        self.fitted = {}
        self.shortlist = None
        self.near = None
        self.near_x = None

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
            sep = separation([(v, k, self.lib[k]) for v, k, _ in scored])
            self.shortlist = [scored[0][1]] if scored and sep < SEP_RATIO else []
            if scored and not self.shortlist:
                key = scored[0][1]
                if (len(self.lib[key]) >= SEED_MIN_GATES
                        and (key.startswith(NEAR_NAMED) or sep < NEAR_SEP)):
                    self.near = key
            log(f'template screen: best {scored[0][1] if scored else None} separation {sep:.2f} -> '
                f'{"fit" if self.shortlist else "near match" if self.near else "no match"}')
            scored = [t for t in scored if t[1] in self.shortlist]
        models = []
        for v, key, x in scored[:FIT_TOP]:
            if time.process_time() > deadline:
                break
            gates = self.lib[key]
            arch = [(g[0], g[1]) for g in gates]
            ops = sim8.fast_ops(arch, n)
            bounds = [(g[2] - TEMPLATE_SLACK, g[3] + TEMPLATE_SLACK) for g in gates]
            x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
            try:
                x1, _ = sim8.fast_fit(ops, x, sub, bounds=bounds, maxiter=200)
                x2, f = sim8.fast_fit(ops, x1, data, bounds=bounds, maxiter=200)
                x3, f3 = sim8.fast_fit(ops, x2, data, maxiter=200)
                if f3 < f:
                    x2, f = x3, f3
                z = sim8.fast_gof(ops, x2, data)
            except Exception as exc:
                log('fit failed', key, exc)
                continue
            self.fitted[key] = x2
            log(f'template {key}: screen {v:.1f} fitted {f:.1f} z={z:.1f}')
            if z >= TEMPLATE_Z and self.shortlist is not None and key in self.shortlist:
                self.shortlist.remove(key)
                if self.near is None and len(gates) >= SEED_MIN_GATES:
                    self.near, self.near_x = key, x2
                    log(f'template {key} rejected at z={z:.1f} -> near match')
            if z < TEMPLATE_Z:
                m = Model('template:' + key, arch, x2)
                m.nll = f
                models.append(m)
        return models

    def fit_near(self, data, n):
        """Bounded, then free, angle fit of the near-match template (a repair seed)."""
        gates = self.lib[self.near]
        arch = [(g[0], tuple(g[1])) for g in gates]
        if self.near_x is not None:
            return arch, self.near_x
        ops = sim8.fast_ops(arch, n)
        bounds = [(g[2] - TEMPLATE_SLACK, g[3] + TEMPLATE_SLACK) for g in gates]
        x = np.array([(g[2] + g[3]) / 2 for g in gates])
        starts = [(x, bounds)]
        if self.near.startswith(('neg:', 'inv:')):
            starts.append((-x, None))
        best = None
        for x0, bnd in starts:
            x1, f1 = sim8.fast_fit(ops, x0, data, bounds=bnd, maxiter=200)
            x2, f2 = sim8.fast_fit(ops, x1, data, maxiter=200)
            for xx, ff in ((x1, f1), (x2, f2)):
                if best is None or ff < best[1]:
                    best = (xx, ff)
        return arch, best[0]


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
                del top[8:]
            sep = separation([(v, k, [(a, t) for a, t in arch]) for v, k, arch, _ in top])
            log(f'generator screened {count} seeds; top {[(round(v), k) for v, k, _, _ in top[:3]]} separation {sep:.2f}')
            cands = top[:1] if sep < SEP_RATIO else []
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
        self.engines = {'p2': pursuit.Pursuit2(n, **kw), 'p1': pursuit.Pursuit(n, **kw),
                        'seed': pursuit.Pursuit2(n, max_gates=rules.patch_max_gates,
                                                 max_ents=rules.patch_max_entanglers)}
        self.arch, self.angles = [], np.zeros(0)
        self.seeded = False

    def restrict(self, qubits):
        for e in self.engines.values():
            e.restrict(qubits)

    def seed(self, data, deadline, arch, angles):
        """Insert/delete repair starting from a fitted near-match template."""
        a2, x2, f2 = self.engines['seed'].run(data, deadline, arch, angles)
        m = Model('pursuit:seeded', a2, x2); m.nll = f2
        log(f'pursuit seeded gates={len(a2)} nll={f2:.1f}')
        self.arch, self.angles, self.seeded = a2, x2, True
        return [m]

    def run(self, data, deadline, fresh=None, warm=True):
        """fresh: engine name for a from-scratch fit ('p1'/'p2') or None."""
        out = []
        now = time.process_time()
        fresh_deadline = now + 0.8 * (deadline - now) if (self.arch and warm) else deadline
        if fresh is not None:
            arch, ang, f = self.engines[fresh].run(data, fresh_deadline)
            if arch:
                m = Model(f'pursuit:{fresh}:fresh', arch, ang); m.nll = f; out.append(m)
                log(f'pursuit {fresh} fresh gates={len(arch)} nll={f:.1f}')
        if warm and self.arch and time.process_time() < deadline:
            eng, name = ('seed', 'pursuit:seeded:warm') if self.seeded else ('p2', 'pursuit:p2:warm')
            a2, x2, f2 = self.engines[eng].run(data, deadline, self.arch, self.angles)
            m = Model(name, a2, x2); m.nll = f2; out.append(m)
            log(f'pursuit warm gates={len(a2)} nll={f2:.1f}')
        if out:
            best = min(out, key=lambda m: m.nll)
            self.arch, self.angles = best.arch, best.angles
            self.seeded = best.source.startswith('pursuit:seeded')
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
            self.settings_cap = rules.max_settings

        def cap(self, settings_cap):
            self.settings_cap = min(rules.max_settings, settings_cap)
            self.settings_remaining = self.settings_cap - len(self.seen)

        def query(self, prep, basis, shots):
            from duelkit.quantum import experiment_index, bitstrings
            self.counter += 1
            basis = ''.join(basis)
            idx = experiment_index(prep, basis, rules.qubits)
            r = client.query(idx, int(shots), f'open-{self.stage}-{self.counter}')
            self.remaining -= int(shots)
            self.seen.add(idx)
            self.settings_remaining = self.settings_cap - len(self.seen)
            c = [r['counts'][b] for b in bitstrings(rules.qubits)]
            self.records.append(dict(prep=list(prep), basis=list(basis), counts=c, shots=int(shots), stage=self.stage))
            return np.array(c, dtype=np.int64)

    n = rules.qubits
    bridge = Bridge()
    panel_rng = np.random.default_rng(20260925)
    labels = ['0', '1', '+', '-', '+i', '-i']

    def spend_random(stage):
        """Spend what is left of this block on new uniformly random product settings."""
        st = client.status()
        rem = int(st['available_now'])
        if rem <= 0:
            return
        left = rules.max_settings - int(st['distinct_settings'])
        stages_left = rules.checkpoints - stage + 1
        k = max(0, min(left // stages_left, rem // RANDOM_MIN_SHOTS))
        if k == 0:
            keys = {}
            for r in bridge.records:
                keys[(tuple(r['prep']), ''.join(r['basis']))] = r
            olds = list(keys.values())
            per = rem // max(1, len(olds))
            for i, r in enumerate(olds):
                sh = per + (1 if i < rem % len(olds) else 0)
                if sh:
                    bridge.query(r['prep'], r['basis'], sh)
            return
        bridge.cap(rules.max_settings)
        per = rem // k
        for i in range(k):
            prep = [labels[j] for j in panel_rng.integers(6, size=n)]
            basis = ''.join(panel_rng.choice(list('XYZ'), n))
            bridge.query(prep, basis, per + (1 if i < rem % k else 0))
        log(f'stage {stage} random panel: {k} new settings x ~{per} shots')
    learner = f2def.Frame2Recovery(n, rules.patch_max_gates, rules.patch_max_entanglers, seed=917)
    templates = TemplateSource()
    try:
        generator = GeneratorSource(rules)
    except Exception as exc:
        log('generator unavailable', exc)
        generator = None
    t_cpu0, t_wall0 = time.process_time(), time.monotonic()

    def used():
        return max(time.process_time() - t_cpu0, time.monotonic() - t_wall0)
    hard = False
    last_source = ''
    seeded_once = False
    f2_useful = True
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
        avail = int(state['available_now'])
        pool = []
        run_f2 = f2_useful
        synth, fitb, fallb = F2_BUDGETS.get(stage, F2_BUDGETS[3])
        if run_f2 and stage > 1:
            f2_cpu = ENCOUNTER_CAP - used() - F2_RESERVE.get(stage, 30.0)
            if f2_cpu < F2_MIN:
                run_f2 = False
                log(f'stage {stage}: f2def skipped, {f2_cpu:.0f}s of CPU left for it')
            else:
                synth, fitb, fallb = min(synth, 0.5 * f2_cpu), min(fitb, 0.3 * f2_cpu), min(fallb, 0.1 * f2_cpu)
        f2def.STAGE_BUDGET[stage] = synth
        f2def.FIT_BUDGET = fitb
        f2def.FALLBACK_BUDGET = fallb
        if stage == 1:
            bridge.remaining = min(avail, F2_STAGE1_SHOTS)
            bridge.cap(F2_STAGE1_SETTINGS)
        else:
            bridge.remaining = avail
            bridge.cap(rules.max_settings)
        try:
            if not run_f2:
                raise StopIteration
            proposed, meta = learner.checkpoint(bridge)
            if stage == 1:
                sup1, _ = subsys.support(bridge.records, n)
                rank_sup = int(meta.get('rank', 0)) - 2 * (n - len(sup1))
                incomplete_frame = meta.get('status') == 'RESIDUAL_ALGEBRA_TOO_LARGE' and rank_sup >= FRAME_RANK
                f2_useful = meta.get('status') != 'RESIDUAL_ALGEBRA_TOO_LARGE' or incomplete_frame
                log(f'stage 1 probe: f2def {meta.get("status")} rank {meta.get("rank")} support {len(sup1)} '
                    f'rank_on_support {rank_sup} -> frame-like {f2_useful}')
                if incomplete_frame:
                    bridge.remaining = int(client.status()['available_now'])
                    bridge.cap(rules.max_settings)
                    proposed, meta = learner.checkpoint(bridge)
                    log(f'stage 1 f2def extended: {meta.get("status")} rank {meta.get("rank")}')
            conv = []
            for g in proposed:
                active = [i for i, a in enumerate(g['pauli']) if a != 'I']
                axes = [g['pauli'][i].lower() for i in active]
                if not 1 <= len(active) <= 2 or len(set(axes)) != 1:
                    raise ValueError('illegal rotation')
                conv.append(G('r' + ''.join(axes), tuple(active), float(g['angle'])))
            pool.append(model_from_patch('f2def:' + str(meta.get('status')), tuple(conv)))
        except StopIteration:
            pass
        except Exception as exc:
            log('f2def failed', type(exc).__name__, str(exc)[:200])
        try:
            spend_random(stage)
        except Exception as exc:
            log('random panel failed', type(exc).__name__, str(exc)[:200])
        try:
            data = sim8.records_to_fast(bridge.records, n)
            if last:
                pool.append(model_from_patch('previous', last))
            pool.append(Model('identity', [], []))
            try:
                sup, _ = subsys.support(bridge.records, n)
                log(f'stage {stage} support {sup}')
                if chaser is not None:
                    chaser.restrict(sup if 0 < len(sup) < n else None)
                if 0 < len(sup) <= SUBSYS_MAX:
                    arch, x, _ = subsys.fit(bridge.records, sup, n, time.process_time() + SUBSYS_BUDGET,
                                            seed=stage)
                    ms = Model(f'subsys:{len(sup)}q', arch, x)
                    ms.score(data)
                    pool.append(ms)
                    log(f'subsys {sup}: nll {ms.nll:.1f} z={ms.gof(data):.1f}')
            except Exception as exc:
                log('subsys failed', type(exc).__name__, str(exc)[:200])
            exact = [m for m in pool if m.arch and m.source.startswith(EXACT_SOURCES)]
            if last_source.startswith(EXACT_SOURCES):
                exact += [m for m in pool if m.source == 'previous']
            early = [m.gof(data) for m in exact if math.isfinite(m.score(data))]
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
            if generator is not None and best_z > GOOD_Z and used() < ENCOUNTER_CAP:
                try:
                    got = generator.run(data, time.process_time() + GEN_BUDGET, stage)
                    pool.extend(got)
                    fits += [(m.gof(data), m) for m in got]
                except Exception as exc:
                    log('generator failed', type(exc).__name__, str(exc)[:200])
            best_z = min((z for z, _ in fits), default=math.inf)
            if (stage <= 2 and not seeded_once and chaser is not None and templates.near and best_z > GOOD_Z
                    and used() < ENCOUNTER_CAP):
                seeded_once = True
                try:
                    arch, x = templates.fit_near(data, n)
                    log(f'near template {templates.near}: seeding pursuit')
                    got = chaser.seed(data, time.process_time() + SEED_BUDGET, arch, x)
                    pool.extend(got)
                    fits += [(m.gof(data), m) for m in got]
                except Exception as exc:
                    log('seeded pursuit failed', type(exc).__name__, str(exc)[:200])
            best_z = min((z for z, _ in fits), default=math.inf)
            log(f'stage {stage} best gof z before pursuit: {best_z:.1f}')
            spent = used()
            warm_only = (not explained and best_z <= GOOD_Z and stage > 1
                         and last_source.startswith('pursuit'))
            if chaser is not None and spent < ENCOUNTER_CAP and (best_z > GOOD_Z or warm_only):
                if warm_only:
                    plan = (None, True, WARM_CHECK_BUDGET)
                elif stage == 1:
                    plan = ('p2', False, PURSUIT_BUDGET[1])
                elif hard and stage < rules.checkpoints:
                    plan = (None, True, HARD_WARM_BUDGET)
                elif hard and last_source.startswith('pursuit:seeded'):
                    plan = (None, True, ENCOUNTER_CAP - spent - REFIT_RESERVE)
                elif hard:
                    plan = ('p1', False, ENCOUNTER_CAP - spent - REFIT_RESERVE)
                else:
                    plan = ('p1', True, PURSUIT_BUDGET.get(stage, 90.0))
                fresh, warm, want = plan
                budget = max(0.0, min(want, ENCOUNTER_CAP - spent - REFIT_RESERVE))
                log(f'stage {stage} pursuit plan fresh={fresh} warm={warm} budget={budget:.0f}s hard={hard}')
                try:
                    pool.extend(chaser.run(data, time.process_time() + budget, fresh=fresh, warm=warm))
                except Exception as exc:
                    log('pursuit failed', type(exc).__name__, str(exc)[:200])
            for m in pool:
                if not math.isfinite(m.nll) or m.source.startswith(('f2def', 'previous', 'identity')):
                    m.score(data)
            pool = [m for m in pool if math.isfinite(m.nll) and legal(m.patch(), rules)]
            pool.sort(key=lambda m: m.bic(data))
            if used() < ENCOUNTER_CAP:
                stale = [m for m in pool if m.arch and m.source.startswith(('previous', 'refit:previous'))]
                for m in stale[:REFIT_TOP]:
                    try:
                        free = free_mask(m.angles)
                        bounds = None if free.all() else [(None, None) if fr else (a, a)
                                                          for fr, a in zip(free, m.angles)]
                        x, f = sim8.fast_fit(sim8.fast_ops(m.arch, n), m.angles, data,
                                             bounds=bounds, maxiter=REFIT_ITER)
                        r = Model('refit:' + m.source.replace('refit:', ''), m.arch, x, k=int(free.sum()))
                        r.score(data)
                        if r.nll < m.nll and legal(r.patch(), rules):
                            pool.append(r)
                    except Exception as exc:
                        log('refit failed', m.source, type(exc).__name__, str(exc)[:120])
                pool.sort(key=lambda m: m.bic(data))
            for m in pool[:5]:
                log(f'stage {stage} {m.source:<40} nll={m.nll:.1f} bic={m.bic(data):.1f} k={m.k}')
            if pool:
                last = pool[0].patch()
                if pool[0].source not in ('previous', 'refit:previous'):
                    last_source = pool[0].source
                log(f'stage {stage} winner {pool[0].source} (lineage {last_source})')
                if stage == 1:
                    z1 = pool[0].gof(data) if pool[0].arch else math.inf
                    hard = z1 > GOOD_Z
                    log(f'stage 1 winner z={z1:.1f} -> hard mode {hard}')
        except Exception as exc:
            log('pool failed', type(exc).__name__, str(exc)[:200])
            if pool and legal(pool[0].patch(), rules):
                last = pool[0].patch()
        log(f'stage {stage} done: encounter CPU {time.process_time() - t_cpu0:.0f}s wall {time.monotonic() - t_wall0:.0f}s')
        client.submit_patch(last, note='opendef')
        client.close_checkpoint()
