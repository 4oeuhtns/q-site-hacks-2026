"""Template-seeded repair: can pursuit fix a few edited gates in a known template?

Edited notebook defaults screen as near-matches (separation ~0.86, between true
matches 0.23-0.56 and non-matches 0.95-0.99) and currently fall through to a
fresh pursuit, which cannot reach 72 gates. Here: fit the nearest template (as
TemplateSource does), then run p2 (insert + delete) warm from it, with the
patch-sized gate cap (108 / 36) so insertions are possible at 72 gates.

    python3 dev/open8/t_seeded.py SPEC_JSON DRAW BUDGET1 BUDGET3
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import sys
import time

import numpy as np

sys.path.insert(0, 'dev/open8')
from offline import *  # noqa: E402,F401
import opendef  # noqa: E402
import pursuit  # noqa: E402
import sim8  # noqa: E402

spec_path, draw, b1, b3 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
spec = dict(family="file", path=spec_path)
circ = get_circ(spec, draw)
lib = json.loads((SOL / 'tlib.json').read_text())
out = dict(spec=Path(spec_path).stem, draw=draw, steps=[])
t0 = time.process_time()


def rec(name, arch, ang, **kw):
    e = eps_model(arch, ang, circ)
    out['steps'].append(dict(step=name, gates=len(arch), eps=round(e, 5), pts=round(points(e), 1),
                             cpu=round(time.process_time() - t0), **kw))


for stage, (shots, S, budget) in enumerate([(32000, 360, b1), (96000, 720, b3)], 1):
    d = simulate(circ, shots=shots, S=S, seed=draw + 10 * stage)
    if stage == 1:
        sub = d.subset(np.argsort(-d.counts.sum(1))[:opendef.SCREEN_SETTINGS])
        scored = []
        for key, gates in lib.items():
            arch = [(g[0], g[1]) for g in gates]
            x = np.array([(g[2] + g[3]) / 2 for g in gates])
            scored.append((sim8.fast_nll(sim8.fast_ops(arch, 8), x, sub), key, gates))
        scored.sort(key=lambda t: t[0])
        sep = opendef.separation(scored)
        key = scored[0][1]
        gates = lib[key]
        arch = [(g[0], tuple(g[1])) for g in gates]
        ops = sim8.fast_ops(arch, 8)
        bounds = [(g[2] - opendef.TEMPLATE_SLACK, g[3] + opendef.TEMPLATE_SLACK) for g in gates]
        x = np.array([(g[2] + g[3]) / 2 for g in gates])
        x1, f1 = sim8.fast_fit(ops, x, d, bounds=bounds, maxiter=200)
        x2, f2 = sim8.fast_fit(ops, x1, d, maxiter=200)
        ang = x2 if f2 < f1 else x1
        rec('template_fit', arch, ang, key=key, sep=round(sep, 3), z=round(sim8.fast_gof(ops, ang, d), 1))
        P = pursuit.Pursuit2(8, max_gates=108, max_ents=36)
    else:
        ang, _ = sim8.fast_fit(sim8.fast_ops(arch, 8), ang, d, maxiter=200)
        rec('refit96k', arch, ang)
    arch, ang, cur = P.run(d, time.process_time() + budget, arch, ang)
    rec(f'p2_seeded_{stage}', arch, ang, stats=dict(P.stats))
print(json.dumps(out), flush=True)
