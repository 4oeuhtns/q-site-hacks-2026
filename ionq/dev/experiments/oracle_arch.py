"""Is near_clifford an INFORMATION trap or a SEARCH trap?

Defender variants, all on the identical 180-setting panel / 12000 shots:
  stock      : SDK Ensemble (grouped)
  arch-truth : knows the true gate ARCHITECTURE, fits angles from truth  (info ceiling)
  arch-rand  : knows the true architecture, fits angles from random starts (landscape test)
If arch-* reaches ~100 and stock does not, the attack is a search trap, so a
stronger searcher defeats it and it is NOT 'generally optimal'.
"""
import sys, json, time
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from scipy.optimize import minimize
from qduel_sdk.rules import Rules
from qduel_sdk.local import LocalSession
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.quantum import experiment_index, bitstrings, infidelity, unitary as squ, G
from duelkit.recovery4.quantum import make_panel, design, Likelihood, circuit_and_jac, unitary as u4
from duelkit.recovery4.recovery import Ensemble
from duelkit.recovery4.adapter import as_gates
from screen import CAND

R = Rules()
PANEL = make_panel(4, 180, 60231)
IDS = [experiment_index(e['prep'], ''.join(e['basis']), 4) for e in PANEL]
KETS, BRAS = design(PANEL)

def labels_of(circ, n=4):
    return [''.join(g.name[-1].upper() if q in g.targets else 'I' for q in range(n)) for g in circ]

def gather(session, shots_total):
    """Sweep the panel block by block, closing checkpoints so shots never cross one."""
    counts = np.zeros((180,16), dtype=np.int64)
    cl = session.client()
    blocks = max(1, shots_total // 4000)
    for b in range(blocks):
        base, extra = divmod(4000, 180)
        for j, idx in enumerate(IDS):
            r = cl.query(idx, base+(j<extra), f'g{b}-{j}')
            counts[j] += np.array([r['counts'][x] for x in bitstrings(4)], dtype=np.int64)
        if b < blocks-1:
            cl.close_checkpoint()
    return counts

def fit_angles(labels, x0, like, iters=400):
    def obj(x):
        u,j,*_ = circuit_and_jac(labels, x, 4)
        v,g = like(u)
        return v, np.real(np.einsum('ij,kij->k', g.conj(), j, optimize=True))
    f = minimize(obj, x0, jac=True, method='L-BFGS-B',
                 bounds=[(-np.pi,np.pi)]*len(labels),
                 options={'maxiter':iters,'gtol':1e-9,'ftol':1e-13,'maxls':40})
    return f.x, f.fun

def eps_of(labels, x, A):
    P = u4([{'pauli':p,'angle':-t} for p,t in zip(labels[::-1], x[::-1])], 4)
    return infidelity(P @ A)

def points(e):
    if e <= 0.001: return 100.0
    if e >= 0.1: return 0.0
    return 100*np.log(0.1/e)/np.log(100.0)

print(f"{'attack':<20}{'seed':>6}{'N':>7}{'stock':>9}{'archTruth':>11}{'archRand':>10}{'rand_hits':>10}")
print("-"*74)
rng = np.random.default_rng(5)
for name in ['near_clifford','public_ladder','interleaved_cycle','star','scrambler']:
    tpl = Template.model_validate(CAND[name])
    for seed in (41, 73):
        circ = instantiate(tpl, seed, R)
        A = squ(circ, 4); labels = labels_of(circ)
        xtrue = np.array([float(g.angle) for g in circ])
        for N in (4000, 12000):
            s = LocalSession(circ, R, seed=seed+100000)
            counts = gather(s, N)
            like = Likelihood(KETS, BRAS, counts)
            # stock learner on the same counts
            ens = Ensemble(block=True, max_gates=18, max_entanglers=6)
            fitted = ens.fit(like)
            pg = as_gates(fitted['block'])
            e_stock = infidelity(squ(pg,4) @ A)
            # architecture known, warm start at truth
            xa,_ = fit_angles(labels, xtrue.copy(), like)
            e_truth = eps_of(labels, xa, A)
            # architecture known, cold random starts
            best = (1e9, None); hits = 0
            for _ in range(12):
                x0 = rng.uniform(-np.pi, np.pi, len(labels))
                xr, fv = fit_angles(labels, x0, like)
                er = eps_of(labels, xr, A)
                if er < 1e-3: hits += 1
                if fv < best[0]: best = (fv, er)
            print(f"{name:<20}{seed:>6}{N:>7}{points(e_stock):>9.1f}{points(e_truth):>11.1f}"
                  f"{points(best[1]):>10.1f}{hits:>7}/12")
