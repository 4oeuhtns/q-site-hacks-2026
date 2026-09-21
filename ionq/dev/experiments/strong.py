"""Does a BETTER SEARCHER (not better measurements) fix the hard attacks?

Stock GatePursuit = one greedy pass, beam 2, ~4 s of a ~180 s budget.
Here: randomized-restart pursuit that spends the headroom. Same 180-setting
panel, same 12000 shots, same counts. Only the search changes.
"""
import sys, time
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
from duelkit.recovery4.recovery import Ensemble
from duelkit.recovery4.adapter import as_gates
from duelkit.recovery4.legacy_learners import DenseMLE, MatrixDistance
from screen import CAND

R = Rules(); PANEL = make_panel(4,180,60231)
IDS = [experiment_index(e['prep'],''.join(e['basis']),4) for e in PANEL]
KETS,BRAS = design(PANEL)
DIC, MATS = dictionary(4)
ENTMASK = np.array([sum(c!='I' for c in p)>1 for p in DIC])

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

def refine(labels,x,like,iters=200):
    def obj(v):
        u,j,*_=circuit_and_jac(labels,v,4); val,g=like(u)
        return val, np.real(np.einsum('ij,kij->k',g.conj(),j,optimize=True))
    return minimize(obj,x,jac=True,method='L-BFGS-B',bounds=[(-np.pi,np.pi)]*len(labels),
                    options={'maxiter':iters,'gtol':2e-7,'ftol':1e-11,'maxls':25})

def greedy(like, rng, max_gates=12, max_ent=4, topk=4, temp=True):
    """Greedy insertion with RANDOMIZED candidate choice among the top-k gradients."""
    labels=[]; x=np.array([])
    ndata=like.total; penalty=np.log(ndata)
    while len(labels)<max_gates:
        u,j,prefix,suffix=circuit_and_jac(labels,x,4)
        base,g=like(u)
        ent=int(sum(ENTMASK[DIC.index(p)] for p in labels))
        cand=[]
        for pos in range(len(labels)+1):
            for p,mat in zip(DIC,MATS):
                if sum(c!='I' for c in p)>1 and ent>=max_ent: continue
                if pos>0 and labels[pos-1]==p: continue
                if pos<len(labels) and labels[pos]==p: continue
                der=np.real(np.vdot(g, suffix[pos]@(-.5j*mat)@prefix[pos]))
                cand.append((abs(der),pos,p,der))
        if not cand: break
        cand.sort(reverse=True)
        seen=set(); pool=[]
        for _,pos,p,der in cand:
            if p in seen: continue
            seen.add(p); pool.append((pos,p,der))
            if len(pool)>=topk: break
        trials=[]
        picks = rng.permutation(len(pool))[:2] if temp else range(min(2,len(pool)))
        for i in picks:
            pos,p,der=pool[i]
            L=labels[:pos]+[p]+labels[pos:]
            v=np.insert(x,pos,-np.sign(der)*rng.uniform(.04,.5))
            f=refine(L,v,like); trials.append((f.fun,L,f.x))
        loss,L,v=min(trials,key=lambda t:t[0])
        if 2*ndata*(base-loss) < penalty: break
        labels,x=L,v
    return labels,np.asarray(x)

def strong_fit(like, budget_s=45, seed=0):
    rng=np.random.default_rng(seed)
    best=(np.inf,[],np.array([])); t0=time.time(); n=0
    while time.time()-t0 < budget_s:
        L,x = greedy(like,rng,temp=(n>0))
        if L:
            f=refine(L,x,like,iters=400); val=f.fun
            if val<best[0]: best=(val,L,f.x)
        n+=1
    return best,n

def eps_from(labels,x,A):
    P=u4([{'pauli':p,'angle':-t} for p,t in zip(labels[::-1],x[::-1])],4)
    return infidelity(P@A)

def pts(e):
    return 100.0 if e<=1e-3 else 0.0 if e>=0.1 else 100*np.log(0.1/e)/np.log(100.0)

print(f"{'attack':<20}{'seed':>5}{'N':>7}{'stock':>8}{'strong':>8}{'restarts':>10}{'eps_strong':>12}")
print("-"*72)
for name in ['near_clifford','public_ladder','interleaved_cycle','star']:
    tpl=Template.model_validate(CAND[name])
    for seed in (41,73):
        circ=instantiate(tpl,seed,R); A=squ(circ,4)
        for N in (4000,12000):
            s=LocalSession(circ,R,seed=seed+100000)
            counts=gather(s,N); like=Likelihood(KETS,BRAS,counts)
            ens=Ensemble(block=True,max_gates=18,max_entanglers=6)
            e_stock=infidelity(squ(as_gates(ens.fit(like)['block']),4)@A)
            (val,L,x),n = strong_fit(like, budget_s=40, seed=seed)
            e_strong = eps_from(L,x,A) if L else 1.0
            print(f"{name:<20}{seed:>5}{N:>7}{pts(e_stock):>8.1f}{pts(e_strong):>8.1f}{n:>10}{e_strong:>12.3e}",flush=True)
