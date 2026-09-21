"""Q1: topology-first identification. Can the coupled pairs be read off the
dense estimate BEFORE searching gate orders, and does restricting the entangler
dictionary to them actually help?

Metric: for each qubit pair (i,j), how much weight does U P_i U^dag put on
operators with support on j (from the PTM). Compare the ranking against the
attack's true entangler pairs.
"""
import sys, math, json, itertools
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from qduel_sdk.rules import Rules
from qduel_sdk.local import LocalSession
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.quantum import experiment_index, bitstrings, unitary as squ, ptm
from duelkit.recovery4.quantum import make_panel, design, Likelihood
from duelkit.recovery4.legacy_learners import DenseMLE
from screen import CAND

R=Rules(); PANEL=make_panel(4,180,60231)
IDS=[experiment_index(e['prep'],''.join(e['basis']),4) for e in PANEL]
KETS,BRAS=design(PANEL)
LAB=[''.join(p) for p in itertools.product('IXYZ',repeat=4)]
PAIRS=list(itertools.combinations(range(4),2))

def gather(session,n):
    counts=np.zeros((180,16),dtype=np.int64); cl=session.client()
    nb=max(1,n//4000)
    for b in range(nb):
        base,extra=divmod(4000,180)
        for j,idx in enumerate(IDS):
            r=cl.query(idx,base+(j<extra),f'g{b}-{j}')
            counts[j]+=np.array([r['counts'][x] for x in bitstrings(4)],dtype=np.int64)
        if b<nb-1: cl.close_checkpoint()
    return counts

def coupling(U):
    """coupling[i,j] = weight U moves from single-qubit Paulis on i onto qubit j."""
    T=ptm(U); C=np.zeros((4,4))
    for i in range(4):
        cols=[k for k,l in enumerate(LAB) if l[i]!='I' and all(l[q]=='I' for q in range(4) if q!=i)]
        for c in cols:
            col=T[:,c]**2
            for r,l in enumerate(LAB):
                if l[i]=='I' and l=='I'*4: continue
                for j in range(4):
                    if j!=i and l[j]!='I': C[i,j]+=col[r]
    return (C+C.T)/2

def true_pairs(circ):
    return {tuple(sorted(g.targets)) for g in circ if len(g.targets)==2}

print(f"{'attack':<20}{'seed':>5}{'true pairs':<22}{'top-4 by coupling':<26}{'hit'}")
print("-"*82)
for name in ['near_clifford','public_ladder','star','interleaved_cycle','scrambler']:
    tpl=Template.model_validate(CAND[name])
    for seed in (41,73):
        circ=instantiate(tpl,seed,R); A=squ(circ,4)
        s=LocalSession(circ,R,seed=seed+100000)
        like=Likelihood(KETS,BRAS,gather(s,12000))
        U=DenseMLE(4,maxiter=500,restarts=2).fit(like)
        C=coupling(U)
        rank=sorted(PAIRS,key=lambda p:-C[p[0],p[1]])
        tp=true_pairs(circ); top=rank[:max(1,len(tp))]
        hit=len(tp & set(top))
        print(f"{name:<20}{seed:>5}{str(sorted(tp)):<22}{str(top):<26}{hit}/{len(tp)}",flush=True)
