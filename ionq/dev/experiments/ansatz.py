"""Can a FIXED 18-gate / 6-entangler ansatz invert an arbitrary legal 4q attack?

If yes, the defender never needs to identify the opponent's architecture: it
fits one fixed circuit shape and submits it. Test with the EXACT attack unitary
(no shot noise), which is an upper bound on what any defender could do with that
ansatz. If it fails here it fails everywhere.
"""
import sys, itertools, time
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from scipy.optimize import minimize
from duelkit.recovery4.quantum import pauli, circuit_and_jac, unitary as u4
from duelkit.quantum import infidelity

PAIRS = [(0,1),(2,3),(1,2),(0,3),(0,2),(1,3)]
LOCAL = [(q,a) for q in range(4) for a in 'XYZ']

def lab(axis, targets):
    return ''.join(axis if q in targets else 'I' for q in range(4))

def brickwork(pair_axes, local_axes):
    """6 entanglers on all 6 pairs + 12 locals = 18 gates, 6 entanglers (at cap)."""
    labels=[]
    for k,(p,ax) in enumerate(zip(PAIRS, pair_axes)):
        a,b = p
        labels.append(lab(local_axes[2*k], (a,)))
        labels.append(lab(local_axes[2*k+1], (b,)))
        labels.append(lab(ax, p))
    return labels

def fit(labels, A, restarts=6, seed=0):
    """min over angles of infidelity(P A) with P = circuit(labels, x)."""
    rng = np.random.default_rng(seed); best = 1.0
    d = 16
    def obj(x):
        U,J,*_ = circuit_and_jac(labels, x, 4)
        M = U @ A
        tr = np.trace(M)
        f = 1 - abs(tr)**2/d**2
        # d/dx_k of |tr(J_k A)| term
        gtr = np.array([np.trace(j @ A) for j in J])
        g = -2*np.real(np.conj(tr)*gtr)/d**2
        return float(np.clip(f,0,1)), g
    for r in range(restarts):
        x0 = rng.uniform(-np.pi, np.pi, len(labels)) if r else np.zeros(len(labels))
        f = minimize(obj, x0, jac=True, method='L-BFGS-B',
                     bounds=[(-np.pi,np.pi)]*len(labels),
                     options={'maxiter':900,'gtol':1e-12,'ftol':1e-15,'maxls':40})
        best = min(best, float(f.fun))
    return best

def random_attack(rng, gates=12, ents=4):
    slots = set(rng.choice(gates, ents, replace=False))
    labels=[]
    for i in range(gates):
        if i in slots:
            p = PAIRS[rng.integers(6)]; labels.append(lab('XYZ'[rng.integers(3)], p))
        else:
            q = int(rng.integers(4)); labels.append(lab('XYZ'[rng.integers(3)], (q,)))
    x = rng.uniform(-np.pi, np.pi, gates)
    return labels, x

rng = np.random.default_rng(2026)
print("A) fixed brickwork ansatz vs random 12-gate/4-entangler attacks (exact unitary)")
print(f"{'trial':>6}{'best_eps_fixed':>17}{'best_eps_axis_search':>23}{'secs':>8}")
for trial in range(5):
    al, ax = random_attack(rng)
    A = u4([{'pauli':p,'angle':t} for p,t in zip(al,ax)], 4)
    t0=time.time()
    fixed = fit(brickwork(['Z']*6, ['X','Y']*6), A, restarts=8, seed=trial)
    best_axis = 1.0
    for pa in itertools.product('XYZ', repeat=3):
        pair_axes = list(pa)+list(pa)
        for la in (['X','Y']*6, ['Y','Z']*6, ['Z','X']*6):
            best_axis = min(best_axis, fit(brickwork(pair_axes, la), A, restarts=3, seed=trial))
    print(f"{trial:>6}{fixed:>17.3e}{best_axis:>23.3e}{time.time()-t0:>8.1f}")

print()
print("B) how much circuit does it take? exact-inverse-architecture sanity + budget sweep")
al, ax = random_attack(rng)
A = u4([{'pauli':p,'angle':t} for p,t in zip(al,ax)], 4)
inv = [{'pauli':p,'angle':-t} for p,t in zip(al[::-1], ax[::-1])]
print("  true inverse architecture eps =", f"{infidelity(u4(inv,4) @ A):.3e}")
for nent in (4,5,6):
    for reps in (1,2):
        labels=[]
        for k in range(nent):
            a,b = PAIRS[k%6]
            labels += [lab('X',(a,)), lab('Y',(b,)), lab('Z',PAIRS[k%6])]
        labels = labels*reps
        if len(labels) > 18 or sum(1 for l in labels if sum(c!='I' for c in l)>1) > 6: continue
        print(f"  ansatz ent={nent} reps={reps} gates={len(labels):>3}  eps={fit(labels,A,restarts=6):.3e}")
