"""Why does near-Clifford trap? Vary the cluster point and width and see.

Prediction from the Clifford-conjugation argument:
  - clustering at +-pi (also Clifford) should trap too
  - tighter clustering -> harder
  - clustering at a NON-Clifford point (e.g. 1.0 rad) should NOT trap
That last one is the falsifier. If 'clustered at 1.0 rad' also traps, the
mechanism is 'narrow ranges' and the Clifford story is wrong.
"""
import sys, math, json
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from harness import run_case, AttackCase, baseline_defender
from qduel_sdk.rules import Rules
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate, sampling_readiness

R = Rules()
P2 = math.pi/2

def frame(center, half, off=(0.8,1.3)):
    """12 gates / 4 entanglers: 8 slots clustered at +-center, 4 off-cluster locals."""
    def c(sign=1):
        lo, hi = sign*center-half, sign*center+half
        lo, hi = max(-math.pi,min(lo,hi)), min(math.pi,max(lo,hi))
        return lo, hi
    g=[]
    for q,a in zip(range(4),'xyzx'):
        lo,hi=c(1 if q%2==0 else -1); g.append({'name':'r'+a,'targets':[q],'low':lo,'high':hi})
    lo,hi=c(1);  g.append({'name':'rzz','targets':[0,1],'low':lo,'high':hi})
    lo,hi=c(-1); g.append({'name':'rxx','targets':[2,3],'low':lo,'high':hi})
    for q,a in zip(range(4),'yzxy'):
        g.append({'name':'r'+a,'targets':[q],'low':off[0],'high':off[1]})
    lo,hi=c(1);  g.append({'name':'ryy','targets':[1,2],'low':lo,'high':hi})
    lo,hi=c(-1); g.append({'name':'rzz','targets':[0,3],'low':lo,'high':hi})
    return g

VAR = {
 'quarter w=.12':  frame(P2, .12),
 'quarter w=.02':  frame(P2, .02),
 'quarter w=.40':  frame(P2, .40),
 'pi      w=.12':  frame(math.pi-.13, .12),
 'noncliff@1.0':   frame(1.00, .12),
 'noncliff@2.2':   frame(2.20, .12),
}
SEEDS=[41,73,907]
print(f"{'variant':<18}{'mean':>8}{'s41':>8}{'s73':>8}{'s907':>8}{'ready':>8}")
print("-"*58)
for name,gates in VAR.items():
    tpl_d={'name':name[:60],'gates':gates}
    try:
        tpl=Template.model_validate(tpl_d); rd=sampling_readiness(tpl,R)['qualifying']
    except ValueError as e:
        print(f"{name:<18} INVALID: {str(e)[:40]}"); continue
    pts=[]
    for s in SEEDS:
        try:
            row=run_case(AttackCase(name,instantiate(tpl,s,R),s,'template',name),R,baseline_defender)
            pts.append(row['recovery_points'])
        except Exception as e:
            pts.append(float('nan'))
    print(f"{name:<18}{np.nanmean(pts):>8.1f}{pts[0]:>8.1f}{pts[1]:>8.1f}{pts[2]:>8.1f}{rd:>8}",flush=True)
