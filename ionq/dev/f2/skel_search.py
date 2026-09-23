"""Hill-climb frame2 skeletons for inverse-compile cost under f2def's synthesis search.

A skeleton is everything but the two insertion angles: per layer, 8 local
(axis, sign) quarter-turns, a perfect matching with pair (axis, sign), and the
two insertion slots with their axes. Cost = cheapest exact-inverse network the
f2def search finds within a short CPU budget, ranked (entanglers, gates). Higher
is harder for any defender whose compiler is at least as good as ours.

    python3 dev/f2/skel_search.py --minutes 45 --workers 7
    python3 dev/f2/skel_search.py --verify dev/results8/skeletons.json --budget 120
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_quantum_duel_sdk_0_7_2'))
sys.path.insert(0, str(ROOT / 'quantum_duel_work/quantum-duel-8q-frame2-0.7/my_solution'))
import numpy as np  # noqa: E402
from duelkit.recovery8.pauli_compile import clifford_absorb  # noqa: E402
import f2def  # noqa: E402

N = 8
AX = 'xyz'
INSERT_ANGLE = 0.8  # representative; compile cost depends on the Paulis, not the angle


def random_skeleton(rng):
    layers = []
    for _ in range(6):
        local = [(str(rng.choice(list(AX))), int(rng.choice([-1, 1]))) for _ in range(8)]
        perm = [int(q) for q in rng.permutation(8)]
        pairs = [(str(rng.choice(list(AX))), int(rng.choice([-1, 1]))) for _ in range(4)]
        layers.append(dict(local=local, perm=perm, pairs=pairs))
    slots = rng.choice(48, 2, replace=False)
    ins = [dict(slot=int(s), axis=str(rng.choice(list(AX))), sign=int(rng.choice([-1, 1]))) for s in slots]
    return dict(layers=layers, ins=ins)


def mutate(sk, rng):
    sk = json.loads(json.dumps(sk))
    kind = rng.integers(5)
    L = sk['layers'][rng.integers(6)]
    if kind == 0:
        q = rng.integers(8); L['local'][q] = (str(rng.choice(list(AX))), int(rng.choice([-1, 1])))
    elif kind == 1:
        i, j = rng.choice(8, 2, replace=False); L['perm'][i], L['perm'][j] = L['perm'][j], L['perm'][i]
    elif kind == 2:
        k = rng.integers(4); L['pairs'][k] = (str(rng.choice(list(AX))), int(rng.choice([-1, 1])))
    elif kind == 3:
        which = rng.integers(2); other = sk['ins'][1 - which]['slot']
        sk['ins'][which]['slot'] = int(rng.choice([s for s in range(48) if s != other]))
    else:
        which = rng.integers(2); sk['ins'][which]['axis'] = str(rng.choice(list(AX)))
    return sk


def circuit(sk, angles=(INSERT_ANGLE, INSERT_ANGLE)):
    """Pauli-dict gate list, chronological."""
    ins = {d['slot']: (d['axis'], d['sign'] * a) for d, a in zip(sk['ins'], angles)}
    out = []
    for li, L in enumerate(sk['layers']):
        for q in range(8):
            if 8 * li + q in ins:
                axis, ang = ins[8 * li + q]
            else:
                axis, s = L['local'][q]; ang = s * math.pi / 2
            out.append({'pauli': ''.join(axis.upper() if k == q else 'I' for k in range(8)), 'angle': ang})
        for k, (axis, s) in enumerate(L['pairs']):
            a, b = L['perm'][2 * k], L['perm'][2 * k + 1]
            out.append({'pauli': ''.join(axis.upper() if x in (a, b) else 'I' for x in range(8)), 'angle': s * math.pi / 2})
    return out


def template(sk, name, lo=0.75, hi=0.9):
    ins = {d['slot']: d for d in sk['ins']}
    gates = []
    for li, L in enumerate(sk['layers']):
        for q in range(8):
            if 8 * li + q in ins:
                d = ins[8 * li + q]
                low, high = (lo, hi) if d['sign'] > 0 else (-hi, -lo)
                gates.append(dict(name='r' + d['axis'], targets=[q], low=low, high=high))
            else:
                axis, s = L['local'][q]
                gates.append(dict(name='r' + axis, targets=[q], low=s * math.pi / 2, high=s * math.pi / 2))
        for k, (axis, s) in enumerate(L['pairs']):
            a, b = sorted((L['perm'][2 * k], L['perm'][2 * k + 1]))
            gates.append(dict(name='r' + axis * 2, targets=[a, b], low=s * math.pi / 2, high=s * math.pi / 2))
    return dict(name=name, gates=gates)


def cost(sk, budget, seed=0):
    inv = f2def.inverse(circuit(sk))
    non, rows = clifford_absorb(inv, [], N, 1e-8)
    best, _ = f2def.search([(non, rows, 0.0)], N, 108, 36, time.process_time() + budget,
                           np.random.default_rng(seed))
    gs = best[1]
    return (f2def.entanglers(gs), len(gs)), len(non)


def climb(args):
    seed, minutes, budget = args
    rng = np.random.default_rng(seed)
    end = time.time() + 60 * minutes
    cur = random_skeleton(rng)
    cur_cost, _ = cost(cur, budget, seed)
    best = (cur_cost, cur)
    evals = 1
    stale = 0
    while time.time() < end:
        cand = mutate(cur, rng)
        if rng.random() < 0.3:
            cand = mutate(cand, rng)
        c, nrot = cost(cand, budget, seed + evals)
        evals += 1
        if nrot != 2:
            continue  # insertions cancelled or merged to Clifford; not a real frame2 attack
        if c >= cur_cost:
            stale = stale + 1 if c == cur_cost else 0
            cur, cur_cost = cand, c
            if c > best[0]:
                best = (c, cand)
        else:
            stale += 1
        if stale > 150:  # restart from the best, perturbed
            cur = mutate(mutate(best[1], rng), rng); cur_cost, _ = cost(cur, budget, seed + evals); stale = 0
    return dict(seed=seed, evals=evals, cost=list(best[0]), skeleton=best[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=45)
    ap.add_argument('--workers', type=int, default=7)
    ap.add_argument('--budget', type=float, default=0.6, help='CPU s per cost eval (climb) or per verify')
    ap.add_argument('--out', default=str(ROOT / 'dev/results8/skeletons.json'))
    ap.add_argument('--verify', default=None)
    args = ap.parse_args()
    from concurrent.futures import ProcessPoolExecutor
    if args.verify:
        found = json.loads(Path(args.verify).read_text())
        jobs = [(r['skeleton'], args.budget, 10_000 + i) for i, r in enumerate(found)]
        with ProcessPoolExecutor(args.workers) as pool:
            res = list(pool.map(_verify, jobs))
        for r, v in zip(found, res):
            r['verified'] = v
            print(r['seed'], 'climb', r['cost'], 'verified', v)
        Path(args.verify).write_text(json.dumps(found, indent=1))
        return
    jobs = [(1000 + i, args.minutes, args.budget) for i in range(args.workers)]
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(climb, jobs))
    res.sort(key=lambda r: r['cost'], reverse=True)
    for r in res:
        print(r['seed'], r['evals'], r['cost'])
    Path(args.out).write_text(json.dumps(res, indent=1))


def _verify(job):
    sk, budget, seed = job
    return list(cost(sk, budget, seed)[0])


if __name__ == '__main__':
    main()
