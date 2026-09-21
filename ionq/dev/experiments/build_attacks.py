"""Build and qualify the two Cup 1 attack templates, then measure them."""
import sys, json, math
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from qduel_sdk.rules import Rules
from qduel_sdk.contracts import Template
from qduel_sdk.templates import qualify, instantiate, sampling_readiness
from harness import run_case, AttackCase, baseline_defender, load_defender

R=Rules(); Q=math.pi/2
HI, LO = 1.69, 1.45          # straddles +pi/2
NHI, NLO = -1.45, -1.69      # straddles -pi/2
OFF = (0.8, 1.3)             # off-Clifford locals

def g(name, targets, lo, hi):
    return {'name':name,'targets':list(targets),'low':lo,'high':hi}

# --- Slot 1: plain quarter-turn frame (the archetype that scored 0.0 everywhere)
A1 = {'name':'Quarter-turn frame','gates':[
    g('rx',[0],LO,HI), g('ry',[1],NHI and NLO, NHI), g('rz',[2],LO,HI), g('rx',[3],NLO,NHI),
    g('rzz',[0,1],LO,HI), g('rxx',[2,3],NLO,NHI),
    g('ry',[0],*OFF), g('rz',[1],*OFF), g('rx',[2],*OFF), g('ry',[3],*OFF),
    g('ryy',[1,2],LO,HI), g('rzz',[0,3],NLO,NHI),
]}
A1['gates'][1] = g('ry',[1],NLO,NHI)

# --- Slot 2: quarter-turn frame whose two insertions MERGE to ~a quarter turn.
# gates 4 and 6 are rz on q0; gate 5 is rzz(0,1), same axis, so it commutes
# through and the pair merges to 2p-0.3 in [1.40,1.60], straddling pi/2.
# Each declared insertion stays off-Clifford; only the effective residual is not.
A2 = {'name':'Merged-insertion quarter-turn frame',
      'parameters':{'p':{'low':0.85,'high':0.95}},
      'gates':[
    {'name':'rz','targets':[0],'low':Q,'high':Q},
    {'name':'rx','targets':[1],'low':-Q,'high':-Q},
    {'name':'ry','targets':[2],'low':Q,'high':Q},
    {'name':'rz','targets':[3],'low':-Q,'high':-Q},
    {'name':'rz','targets':[0],'parameter':'p','scale':1.0,'offset':0.0},
    {'name':'rzz','targets':[0,1],'low':Q,'high':Q},
    {'name':'rz','targets':[0],'parameter':'p','scale':1.0,'offset':-0.3},
    {'name':'rxx','targets':[2,3],'low':-Q,'high':-Q},
    {'name':'ry','targets':[1],'low':Q,'high':Q},
    {'name':'rx','targets':[2],'low':-Q,'high':-Q},
    {'name':'ryy','targets':[1,2],'low':Q,'high':Q},
    {'name':'rzz','targets':[0,3],'low':-Q,'high':-Q},
]}

templates=[Template.model_validate(A1), Template.model_validate(A2)]
for t in templates:
    ent=sum(1 for x in t.gates if len(x.targets)==2)
    print(f"{t.name:<40} gates={len(t.gates):>3} entanglers={ent}")
    print("   readiness:", sampling_readiness(t,R))

print("\nqualify():")
try:
    print(" ", {k:v for k,v in qualify(templates,R).items() if k in ('status','ruleset','qubits')})
except ValueError as e:
    print("  FAILED:", e); raise SystemExit(1)

MINE="/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/my_solution/main.py"
mine=load_defender(MINE)
SEEDS=[41,73,907,1234,5150]
print(f"\n{'template':<40}{'defender':<10}{'mean':>7}{'worst':>7}  per-seed")
for t,payload in zip(templates,(A1,A2)):
    for label,d in (('stock',baseline_defender),('mine',mine)):
        pts=[]
        for s in SEEDS:
            row=run_case(AttackCase(t.name,instantiate(t,s,R),s,'template',t.name),R,d)
            pts.append(row['recovery_points'])
        print(f"{t.name:<40}{label:<10}{np.mean(pts):>7.1f}{max(pts):>7.1f}  "
              f"{['%.0f'%p for p in pts]}",flush=True)

out="/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/attacks.json"
json.dump([t.model_dump() for t in templates], open(out,'w'), indent=2)
print("\nwrote", out)
