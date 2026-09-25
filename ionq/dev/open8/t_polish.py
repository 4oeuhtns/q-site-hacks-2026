"""p3 then polish, within the same CPU budget t_frontier gives p1.

p3 (insert-only) stops early on BIC with extra gates and a near-miss fit. Measure
what the leftover budget buys: a long refit, stepwise deletion, then a warm p2
(insert + delete) continuation.

    python3 dev/open8/t_polish.py G SEED DRAW SHOTS BUDGET
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, 'dev/open8')
from offline import *  # noqa: E402,F401
import pursuit  # noqa: E402
import sim8  # noqa: E402

g, seed, draw, shots, budget = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
spec = dict(family="generic", seed=seed, gates=g, ents=g // 3, bands=[[0.6, 1.2], [1.9, 2.5]])
circ = get_circ(spec, draw)
d = simulate(circ, shots=shots, S={32000: 360, 64000: 540, 96000: 720}.get(shots, 720), seed=draw)
t0 = time.process_time()
end = t0 + budget
steps = []


def rec(name, arch, ang):
    e = eps_model(arch, ang, circ)
    steps.append(dict(step=name, gates=len(arch), eps=round(e, 5), pts=round(points(e), 1),
                      cpu=round(time.process_time() - t0)))


P3 = pursuit.Pursuit3(8, max_gates=72, max_ents=24)
arch, ang, cur = P3.run(d, end)
rec('p3', arch, ang)
if arch and time.process_time() < end:
    ang, cur = sim8.fast_fit(sim8.fast_ops(arch, 8), ang, d, maxiter=1000)
    rec('refit1000', arch, ang)
P2 = pursuit.Pursuit2(8, max_gates=72, max_ents=24)
pen = math.log(max(d.N, 2))
for _ in range(30):
    if len(arch) < 2 or time.process_time() > end:
        break
    n_before = len(arch)
    arch, ang, cur = P2._eliminate(arch, ang, d, cur, pen, end)
    if len(arch) == n_before:
        break
rec('eliminate', arch, ang)
if time.process_time() < end:
    arch, ang, cur = P2.run(d, end, arch, ang)
    rec('p2warm', arch, ang)
print(json.dumps(dict(g=g, seed=seed, draw=draw, shots=shots, budget=budget, steps=steps)), flush=True)
