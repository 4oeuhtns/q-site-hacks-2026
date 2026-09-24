"""Offline learner test: random product panel -> counts -> learner -> eps vs truth."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, json, math, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # sets sys.path for SDK
SOL = HERE.parents[1] / "quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution"
sys.path.insert(0, str(SOL))
import sim8
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.quantum import unitary, infidelity

LAB = ['0', '1', '+', '-', '+i', '-i']

def panel(rng, S, n=8):
    preps = [[LAB[i] for i in rng.integers(6, size=n)] for _ in range(S)]
    bases = [''.join(rng.choice(list('XYZ'), n)) for _ in range(S)]
    return preps, bases

def simulate(circ, shots=32000, S=360, seed=0, n=8):
    rng = np.random.default_rng(seed)
    preps, bases = panel(rng, S, n)
    ops = sim8.fast_ops([(g.name, g.targets) for g in circ], n)
    d = sim8.FastData(preps, bases, np.zeros((S, 1 << n)), n)
    st = sim8.fast_state(ops, [g.angle for g in circ], d)
    sim8._readout(st, d.mats, n)
    P = st.real ** 2 + st.imag ** 2
    per = np.full(S, shots // S); per[: shots % S] += 1
    counts = np.stack([rng.multinomial(k, p / p.sum()) for k, p in zip(per, P)])
    return sim8.FastData(preps, bases, counts, n)

def eps_model(arch, angles, circ, n=8):
    from duelkit.quantum import G
    Ua = unitary(circ, n)
    Um = unitary(tuple(G(a, t, float(x)) for (a, t), x in zip(arch, angles)), n) if arch else np.eye(1 << n)
    return infidelity(Um.conj().T @ Ua)

def points(e):
    return 100.0 if e <= 1e-3 else 0.0 if e >= 0.1 else 100 * math.log(0.1 / e) / math.log(100)

def get_circ(spec, draw):
    return instantiate(Template.model_validate(lab.spec_to_template(spec)), draw, lab.RULES)
