"""Rebuild attacks.json: BOTH slots ranged near-quarter-turn frames.

The merged-insertion template fell to the ported frame learner (100/94/100)
because its frame gates sit at EXACTLY +-pi/2, which ParityLearner identifies
exactly. Ranged angles straddling pi/2 are never exactly Clifford, so the
surviving Pauli relations do not hold and the algebraic route gets nothing.
"""
import sys, json, math
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from qduel_sdk.rules import Rules
from qduel_sdk.contracts import Template
from qduel_sdk.templates import qualify, instantiate, sampling_readiness
from harness import run_case, AttackCase, baseline_defender, load_defender
from frame4 import frame_defender

R=Rules()
LO,HI   = 1.45, 1.69     # straddles +pi/2, never equals it
NLO,NHI = -1.69, -1.45   # straddles -pi/2
OFF     = (0.8, 1.3)

def g(name,targets,lo,hi): return {'name':name,'targets':list(targets),'low':lo,'high':hi}

A1={'name':'Ranged quarter-turn frame, cycle','gates':[
    g('rx',[0],LO,HI), g('ry',[1],NLO,NHI), g('rz',[2],LO,HI), g('rx',[3],NLO,NHI),
    g('rzz',[0,1],LO,HI), g('rxx',[2,3],NLO,NHI),
    g('ry',[0],*OFF), g('rz',[1],*OFF), g('rx',[2],*OFF), g('ry',[3],*OFF),
    g('ryy',[1,2],LO,HI), g('rzz',[0,3],NLO,NHI),
]}

A2={'name':'Ranged quarter-turn frame, crossed','gates':[
    g('ry',[0],LO,HI), g('rz',[1],NLO,NHI), g('rx',[2],LO,HI), g('ry',[3],LO,HI),
    g('rxx',[0,2],LO,HI), g('rzz',[1,3],NLO,NHI),
    g('rz',[0],*OFF), g('rx',[1],*OFF), g('ry',[2],*OFF), g('rz',[3],*OFF),
    g('ryy',[0,1],LO,HI), g('rxx',[2,3],NLO,NHI),
]}

templates=[Template.model_validate(A1),Template.model_validate(A2)]
for t in templates:
    ent=sum(1 for x in t.gates if len(x.targets)==2)
    print(f"{t.name:<38} gates={len(t.gates)} ent={ent} readiness={sampling_readiness(t,R)['qualifying']}/64")
print("qualify:", {k:v for k,v in qualify(templates,R).items() if k=='status'})

mine=load_defender('/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/my_solution/main.py')
SEEDS=[41,73,907,1234]
print(f"\n{'template':<38}{'defender':<9}{'mean':>7}{'worst':>7}  per-seed")
for t in templates:
    for label,d in (('stock',baseline_defender),('frame4',frame_defender),('mine',mine)):
        pts=[run_case(AttackCase(t.name,instantiate(t,s,R),s,'template',t.name),R,d)['recovery_points']
             for s in SEEDS]
        print(f"{t.name:<38}{label:<9}{np.mean(pts):>7.1f}{max(pts):>7.1f}  {['%.0f'%p for p in pts]}",flush=True)

out='/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/attacks.json'
json.dump([t.model_dump() for t in templates],open(out,'w'),indent=2)
print("\nwrote",out)
