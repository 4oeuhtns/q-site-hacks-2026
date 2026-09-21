"""Q2: does a Clifford-aware search crack near_clifford?

archRand showed 0/12 random angle starts find the basin even with the true
architecture. But the true angles cluster at +-pi/2. Stock GatePursuit inserts
every new gate at angle ~0.08. So the search literally never looks where the
answer is. Fix: seed each insertion at {+pi/2, -pi/2, small}, and add a
'snap every angle to the nearest quarter turn and re-refine' move.
"""
import sys, time, math
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from scipy.optimize import minimize
from qduel_sdk.rules import Rules
from qduel_sdk.local import LocalSession
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.quantum import experiment_index, bitstrings, infidelity, unitary as squ
from duelkit.recovery4.quantum import (make_panel, design, Likelihood, circuit_and_jac,
                                       unitary as u4, dictionary)
from duelkit.recovery4.recovery import Ensemble, deviance
from duelkit.recovery4.adapter import as_gates
from duelkit.recovery4.legacy_learners import DenseMLE, GatePursuit, MatrixDistance
from screen import CAND

R=Rules(); PANEL=make_panel(4,180,60231)
IDS=[experiment_index(e['prep'],''.join(e['basis']),4) for e in PANEL]
KETS,BRAS=design(PANEL); DIC,MATS=dictionary(4)
P2=math.pi/2
SNAP=np.array([-math.pi,-P2,0.0,P2,math.pi])

def gather(session,n):
    counts=np.zeros((180,16),dtype=np.int64); cl=session.client()
    for b in range(max(1,n//4000)):
        base,extra=divmod(4000,180)
        for j,idx in enumerate(IDS):
            r=cl.query(idx,base+(j<extra),f'g{b}-{j}')
            counts[j]+=np.array([r['counts'][x] for x in bitstrings(4)],dtype=np.int64)
        if b < n//4000-1: cl.close_checkpoint()
    return counts

def _obj(x,labels,like):
    u,j,*_=circuit_and_jac(labels,x,4); v,g=like(u)
    return v,np.real(np.einsum('ij,kij->k',g.conj(),j,optimize=True))

def refine(labels,x,like,iters=180):
    return minimize(_obj,x,args=(labels,like),jac=True,method='L-BFGS-B',
                    bounds=[(-math.pi,math.pi)]*len(labels),
                    options={'maxiter':iters,'gtol':2e-7,'ftol':1e-11,'maxls':25})

def cliff_pursuit(like, max_gates=18, max_ent=6, beam=2, seeds=(P2,-P2,0.08)):
    """Greedy insertion, but each candidate is tried from several seed angles."""
    labels=[]; x=np.array([]); ndata=like.total; penalty=math.log(ndata)
    while len(labels)<max_gates:
        u,j,prefix,suffix=circuit_and_jac(labels,x,4)
        base,g=like(u)
        ent=sum(sum(c!='I' for c in p)>1 for p in labels)
        cands=[]
        for pos in range(len(labels)+1):
            for p,mat in zip(DIC,MATS):
                if sum(c!='I' for c in p)>1 and ent>=max_ent: continue
                if pos>0 and labels[pos-1]==p: continue
                if pos<len(labels) and labels[pos]==p: continue
                der=np.real(np.vdot(g,suffix[pos]@(-.5j*mat)@prefix[pos]))
                cands.append((abs(der),pos,p,der))
        cands.sort(reverse=True); trials=[]; seen=set()
        for _,pos,p,der in cands:
            if p in seen: continue
            seen.add(p); L=labels[:pos]+[p]+labels[pos:]
            for s in seeds:
                a = -np.sign(der)*abs(s) if s < 0.5 else s
                f=refine(L,np.insert(x,pos,a),like)
                trials.append((f.fun,L,f.x))
            if len(seen)>=beam: break
        if not trials: break
        loss,L,v=min(trials,key=lambda t:t[0])
        if 2*ndata*(base-loss)<penalty: break
        labels,x=L,v
    return labels,np.asarray(x)

def snap_move(labels,x,like):
    """Snap each angle to the nearest quarter-turn lattice point, re-refine, keep if better."""
    best=(refine(labels,x,like).fun, x.copy())
    xs=SNAP[np.argmin(np.abs(x[:,None]-SNAP[None,:]),axis=1)]
    f=refine(labels,xs,like)
    if f.fun<best[0]: best=(f.fun,f.x)
    for i in range(len(x)):
        xi=x.copy(); xi[i]=SNAP[np.argmin(np.abs(x[i]-SNAP))]
        f=refine(labels,xi,like)
        if f.fun<best[0]: best=(f.fun,f.x)
    return best

def eps_of(L,x,A):
    P=u4([{'pauli':p,'angle':-t} for p,t in zip(L[::-1],x[::-1])],4)
    return infidelity(P@A)
def pts(e): return 100.0 if e<=1e-3 else 0.0 if e>=0.1 else 100*math.log(0.1/e)/math.log(100.0)

print(f"{'attack':<20}{'seed':>5}{'N':>7}{'stock':>8}{'cliff':>8}{'secs':>7}{'eps_cliff':>12}")
print("-"*70)
for name in ['near_clifford','public_ladder','interleaved_cycle','star','scrambler']:
    tpl=Template.model_validate(CAND[name])
    for seed in (41,73,907):
        circ=instantiate(tpl,seed,R); A=squ(circ,4)
        for N in (4000,12000):
            s=LocalSession(circ,R,seed=seed+100000)
            like=Likelihood(KETS,BRAS,gather(s,N))
            ens=Ensemble(block=True,max_gates=18,max_entanglers=6)
            e_stock=infidelity(squ(as_gates(ens.fit(like)['block']),4)@A)
            t0=time.time()
            L,x=cliff_pursuit(like)
            if L:
                v,x2=snap_move(L,x,like); e_c=eps_of(L,x2,A)
            else: e_c=1.0
            print(f"{name:<20}{seed:>5}{N:>7}{pts(e_stock):>8.1f}{pts(e_c):>8.1f}"
                  f"{time.time()-t0:>7.1f}{e_c:>12.3e}",flush=True)
