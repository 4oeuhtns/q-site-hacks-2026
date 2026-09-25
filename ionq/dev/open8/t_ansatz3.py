import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, math, json
sys.path.insert(0,'dev/open8')
from offline import *
from duelkit.quantum import G, unitary, infidelity
from scipy.optimize import minimize

def ansatz3(blocks):
    arch=[]
    for q in range(3): arch += [('rz',(q,)),('ry',(q,)),('rz',(q,))]
    pairs=[(0,1),(1,2),(0,2)]; axes=['zz','xx','yy']
    for b in range(blocks):
        a,c=pairs[b%3]; ax=axes[(b//3)%3]
        arch += [('r'+ax,(a,c)),('ry',(a,)),('rz',(a,)),('ry',(c,)),('rz',(c,))]
    return arch

_P={'x':np.array([[0,1],[1,0]],complex),'y':np.array([[0,-1j],[1j,0]]),'z':np.diag([1,-1]).astype(complex)}
_cache={}
def pauli(name,t,n):
    key=(name,tuple(t),n)
    if key not in _cache:
        ops=[np.eye(2,dtype=complex)]*n; ops=list(ops)
        for q in t: ops[q]=_P[name[-1]]
        M=ops[0]
        for o in ops[1:]: M=np.kron(M,o)
        _cache[key]=M
    return _cache[key]
def U_of(arch, x, n=3):
    U=np.eye(1<<n,dtype=complex)
    for (a,t),v in zip(arch,x):
        U=(math.cos(v/2)*np.eye(1<<n)-1j*math.sin(v/2)*pauli(a,t,n))@U
    return U
def infidelity(M):
    return 1-abs(np.trace(M))**2/M.shape[0]**2

def fit_exact(arch, target, seed, iters=400):
    rng=np.random.default_rng(seed)
    def f(x):
        return infidelity(U_of(arch,x).conj().T @ target)
    x0=rng.uniform(-math.pi,math.pi,len(arch))
    r=minimize(f,x0,method='L-BFGS-B',options=dict(maxiter=iters))
    return r.fun
# random deep 3-qubit attack (72 gates, 24 ent) mapped onto qubits 0..2
spec=dict(family="generic",seed=41,bands=[[0.6,1.2],[1.9,2.5]],qubits=[0,1,2])
for blocks in (14,19):
    arch=ansatz3(blocks)
    for s,dr in ((41,1),(42,2)):
        spec['seed']=s
        circ=get_circ(spec,dr)
        target=U_of([(g.name,g.targets) for g in circ],[g.angle for g in circ])
        t=time.time(); res=[fit_exact(arch,target,seed) for seed in range(3)]
        print(f"blocks={blocks} gates={len(arch)} ents={blocks} params={len(arch)} attack seed {s}: best eps {min(res):.2e} all {[f'{r:.1e}' for r in res]} ({time.time()-t:.0f}s)",flush=True)
