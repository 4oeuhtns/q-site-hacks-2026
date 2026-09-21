"""Cramer-Rao floor on process infidelity for a 4-qubit Quantum Duel encounter.

Assumes the defender knows the ATTACK ARCHITECTURE and only fits angles.
That is an upper bound on defender performance from measurement design alone,
so it separates "design/statistics" from "architecture search".
"""
import sys, itertools, json
import numpy as np
sys.path.insert(0, "/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")

from duelkit.recovery4.quantum import (pauli, rotation, unitary as u4, KETS, BASES, kron,
                                       make_panel, design, circuit_and_jac, dictionary)
from duelkit.quantum import G
from qduel_sdk.contracts import Template
from qduel_sdk.rules import Rules
from qduel_sdk.templates import instantiate

RULES = Rules()
PREPS = ['0','1','+','-','+i','-i']
AXES = ['X','Y','Z']

def circ_to_labels(circuit, n=4):
    labels, angles = [], []
    for g in circuit:
        labels.append(''.join(g.name[-1].upper() if q in g.targets else 'I' for q in range(n)))
        angles.append(float(g.angle))
    return labels, np.array(angles)

def eps_hessian(labels, x):
    """eps(theta_hat) ~ 0.5 d^T M d for patch = inverse(circuit(theta_hat))."""
    d = 16
    def eps(dx):
        Uh = u4([{'pauli':p,'angle':t} for p,t in zip(labels, x+dx)], 4)
        Ut = u4([{'pauli':p,'angle':t} for p,t in zip(labels, x)], 4)
        E = Uh.conj().T @ Ut
        return float(np.clip(1-abs(np.trace(E))**2/d**2, 0, 1))
    k = len(x); h = 1e-4; M = np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            e = np.zeros(k); f = np.zeros(k); e[i]=h; f[j]=h
            M[i,j] = (eps(e+f)-eps(e-f)-eps(-e+f)+eps(-e-f))/(4*h*h)
    return (M+M.T)/2

def fisher(labels, x, panel, analysis_labels=(), analysis_x=()):
    """Per-shot Fisher information matrix averaged uniformly over the panel."""
    U, J, *_ = circuit_and_jac(labels, x, 4)
    if analysis_labels:
        V = u4([{'pauli':p,'angle':t} for p,t in zip(analysis_labels, analysis_x)], 4)
        U = V @ U; J = np.einsum('ab,kbc->kac', V, J)
    kets, bras = design(panel)
    amp = np.einsum('sij,sj->si', bras, kets @ U.T, optimize=True)         # (S,16)
    damp = np.einsum('sij,kja,sa->ksi', bras, J, kets, optimize=True)      # (K,S,16)
    p = np.abs(amp)**2
    dp = 2*np.real(np.conj(amp)[None,:,:]*damp)
    p = np.maximum(p, 1e-12)
    F = np.einsum('ksi,lsi,si->kl', dp, dp, 1/p, optimize=True)/len(panel)
    return F

def floor(M, F, N, ridge=1e-9):
    F = F + ridge*np.eye(len(F))
    return 0.5*np.trace(M @ np.linalg.inv(N*F))

def points(e):
    if e <= 0.001: return 100.0
    if e >= 0.1: return 0.0
    return 100*np.log(0.1/e)/np.log(100.0)

def rand_panel(size, seed, prep_pool=PREPS, axis_pool=AXES):
    rng = np.random.default_rng(seed); seen=set(); out=[]
    while len(out) < size:
        pr = tuple(rng.choice(prep_pool, 4)); ba = tuple(rng.choice(axis_pool, 4))
        if (pr,ba) in seen: continue
        seen.add((pr,ba)); out.append({'prep':list(pr),'basis':list(ba)})
    return out

BANK = json.loads(open('/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev/attacks/4q_ladder.json').read())

def load(name):
    return json.loads(open(f'/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev/attacks/{name}.json').read())

def report(name, seed=41):
    tpl = Template.model_validate(load(name))
    circ = instantiate(tpl, seed, RULES)
    labels, x = circ_to_labels(circ)
    M = eps_hessian(labels, x)
    inv_labels = labels[::-1]; inv_x = -x[::-1]
    rows = []
    designs = {
      'baseline panel-180 (open loop)':   (make_panel(4,180,60231), (), ()),
      'random panel-180 (open loop)':     (rand_panel(180, 7), (), ()),
      'closed loop: analysis = inverse':  (rand_panel(180, 7), inv_labels, inv_x),
      'closed loop, prep-matched basis':  (None, inv_labels, inv_x),
    }
    for label,(panel, al, ax) in designs.items():
        if panel is None:
            rng = np.random.default_rng(11); panel=[]; seen=set()
            while len(panel) < 180:
                pr = tuple(rng.choice(PREPS,4))
                ba = tuple({'0':'Z','1':'Z','+':'X','-':'X','+i':'Y','-i':'Y'}[a] for a in pr)
                if (pr,ba) in seen: continue
                seen.add((pr,ba)); panel.append({'prep':list(pr),'basis':list(ba)})
        F = fisher(labels, x, panel, al, ax)
        ev = np.linalg.eigvalsh(F)
        for N in (4000, 12000):
            e = floor(M, F, N)
            rows.append((label, N, e, points(e), ev.min(), ev.max()))
    print(f"\n=== {name}@{seed}  gates={len(labels)}  params={len(x)} ===")
    print(f"{'design':<34}{'N':>7}{'eps_CRB':>12}{'pts':>8}{'Fmin':>10}{'Fmax':>10}")
    for label,N,e,p,mn,mx in rows:
        print(f"{label:<34}{N:>7}{e:>12.3e}{p:>8.1f}{mn:>10.2e}{mx:>10.2e}")

for nm in ['4q_local','4q_zz','4q_conjugated','4q_mixed','4q_ladder','4q_commutator']:
    report(nm)
