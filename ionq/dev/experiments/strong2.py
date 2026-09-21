"""Corrected: restarts AROUND the dense->project pipeline (not instead of it).

strong.py omitted DenseMLE and lost everywhere, which showed the dense relaxation
is load-bearing. Here the restart budget is spent on:
  - multiple DenseMLE random restarts  (different dense targets)
  - multiple GatePursuit projections of each target
  - a delete/swap/axis local-search pass that stock cannot do (insert-only)
Selection by BIC on the real count likelihood, same 180-panel data as stock.
"""
import sys, time
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
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

R=Rules(); PANEL=make_panel(4,180,60231)
IDS=[experiment_index(e['prep'],''.join(e['basis']),4) for e in PANEL]
KETS,BRAS=design(PANEL); DIC,_=dictionary(4)

def gather(session, shots_total):
    counts=np.zeros((180,16),dtype=np.int64); cl=session.client()
    blocks=max(1,shots_total//4000)
    for b in range(blocks):
        base,extra=divmod(4000,180)
        for j,idx in enumerate(IDS):
            r=cl.query(idx,base+(j<extra),f'g{b}-{j}')
            counts[j]+=np.array([r['counts'][x] for x in bitstrings(4)],dtype=np.int64)
        if b<blocks-1: cl.close_checkpoint()
    return counts

def local_search(gp, like, rounds=2):
    """Moves stock GatePursuit cannot make: delete, swap adjacent, change axis."""
    best=(like(circuit_and_jac(gp.labels,gp.x,4)[0])[0], list(gp.labels), gp.x.copy())
    for _ in range(rounds):
        cur_v,cur_L,cur_x=best; improved=False
        for i in range(len(cur_L)):
            cands=[]
            cands.append((cur_L[:i]+cur_L[i+1:], np.delete(cur_x,i)))          # delete
            if i+1<len(cur_L):                                                  # swap
                L=list(cur_L); L[i],L[i+1]=L[i+1],L[i]
                x=cur_x.copy(); x[i],x[i+1]=x[i+1],x[i]; cands.append((L,x))
            base=cur_L[i]; targets=[q for q,c in enumerate(base) if c!='I']
            for ax in 'XYZ':                                                    # axis
                lab=''.join(ax if q in targets else 'I' for q in range(4))
                if lab!=base:
                    L=list(cur_L); L[i]=lab; cands.append((L,cur_x.copy()))
            for L,x in cands:
                if not L: continue
                f=gp.refine(L,x,like,maxiter=120)
                if f.fun < best[0]-1e-9: best=(f.fun,L,f.x); improved=True
        if not improved: break
    return best

def strong_fit(like, budget_s=40, seed=0):
    rng=np.random.default_rng(seed); t0=time.time(); options=[]; n=0
    while time.time()-t0 < budget_s:
        dense=DenseMLE(4, maxiter=400, restarts=1)
        dense.x = None if n else None
        if n: dense.x = rng.normal(0,.05,len(dense.ps))     # different random basin
        target=dense.fit(like)
        gp=GatePursuit(4,max_gates=18,max_entanglers=6,beam=2)
        gp.fit(MatrixDistance(target, like.total))
        if not gp.labels: n+=1; continue
        f=gp.refine(gp.labels,gp.x,like); gp.x=f.x
        v,L,x = local_search(gp, like)
        options.append((deviance(like, circuit_and_jac(L,x,4)[0], len(L)), L, x))
        n+=1
    if not options: return None, n
    return min(options,key=lambda t:t[0]), n

def eps_from(L,x,A):
    P=u4([{'pauli':p,'angle':-t} for p,t in zip(L[::-1],x[::-1])],4)
    return infidelity(P@A)

def pts(e): return 100.0 if e<=1e-3 else 0.0 if e>=0.1 else 100*np.log(0.1/e)/np.log(100.0)

from screen import CAND
print(f"{'attack':<20}{'seed':>5}{'N':>7}{'stock':>8}{'strong2':>9}{'iters':>7}{'eps':>12}")
print("-"*70)
for name in ['near_clifford','public_ladder','interleaved_cycle','star']:
    tpl=Template.model_validate(CAND[name])
    for seed in (41,73):
        circ=instantiate(tpl,seed,R); A=squ(circ,4)
        for N in (4000,12000):
            s=LocalSession(circ,R,seed=seed+100000)
            like=Likelihood(KETS,BRAS,gather(s,N))
            ens=Ensemble(block=True,max_gates=18,max_entanglers=6)
            e_stock=infidelity(squ(as_gates(ens.fit(like)['block']),4)@A)
            res,n=strong_fit(like,budget_s=40,seed=seed)
            e2 = eps_from(res[1],res[2],A) if res else 1.0
            print(f"{name:<20}{seed:>5}{N:>7}{pts(e_stock):>8.1f}{pts(e2):>9.1f}{n:>7}{e2:>12.3e}",flush=True)
