"""Confirm the deceptive-landscape mechanism: does greedy pursuit stop early on
near_clifford because no SINGLE insertion is BIC-significant?"""
import sys, math
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from qduel_sdk.rules import Rules
from qduel_sdk.local import LocalSession
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.quantum import experiment_index, bitstrings, unitary as squ
from duelkit.recovery4.quantum import make_panel, design, Likelihood, circuit_and_jac
from duelkit.recovery4.legacy_learners import GatePursuit, DenseMLE, MatrixDistance
from screen import CAND

R=Rules(); PANEL=make_panel(4,180,60231)
IDS=[experiment_index(e['prep'],''.join(e['basis']),4) for e in PANEL]
KETS,BRAS=design(PANEL)

def gather(session,n):
    c=np.zeros((180,16),dtype=np.int64); cl=session.client(); nb=max(1,n//4000)
    for b in range(nb):
        base,extra=divmod(4000,180)
        for j,idx in enumerate(IDS):
            r=cl.query(idx,base+(j<extra),f'g{b}-{j}')
            c[j]+=np.array([r['counts'][x] for x in bitstrings(4)],dtype=np.int64)
        if b<nb-1: cl.close_checkpoint()
    return c

print(f"{'attack':<20}{'seed':>5}{'true_gates':>11}{'direct_gates':>14}{'proj_gates':>12}{'first_gain':>12}{'BIC_pen':>9}")
print("-"*86)
for name in ['near_clifford','public_ladder','scrambler','star']:
    tpl=Template.model_validate(CAND[name])
    for seed in (41,73):
        circ=instantiate(tpl,seed,R)
        s=LocalSession(circ,R,seed=seed+100000)
        like=Likelihood(KETS,BRAS,gather(s,12000))
        gp=GatePursuit(4,max_gates=18,max_entanglers=6,beam=2); gp.fit(like)
        dense=DenseMLE(4,maxiter=500,restarts=2).fit(like)
        pr=GatePursuit(4,max_gates=18,max_entanglers=6,beam=2)
        pr.fit(MatrixDistance(dense,like.total))
        # gain from the single best first insertion, vs the BIC penalty
        empty=GatePursuit(4,max_gates=1,max_entanglers=6,beam=2); empty.fit(like)
        base=like(np.eye(16,dtype=complex))[0]
        if empty.labels:
            u,*_=circuit_and_jac(empty.labels,empty.x,4); g1=2*like.total*(base-like(u)[0])
        else: g1=0.0
        print(f"{name:<20}{seed:>5}{len(circ):>11}{len(gp.labels):>14}{len(pr.labels):>12}"
              f"{g1:>12.1f}{math.log(like.total):>9.2f}",flush=True)
