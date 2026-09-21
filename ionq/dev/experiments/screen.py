"""Screen candidate Cup-1 attack archetypes against the stock SDK baseline defender."""
import sys, json, math, time
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from harness import run_case, AttackCase, baseline_defender
from qduel_sdk.rules import Rules
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate, qualify, sampling_readiness

R = Rules()

def T(name, gates, parameters=None):
    d = {'name': name, 'gates': gates}
    if parameters: d['parameters'] = parameters
    return d

def g(n, t, lo, hi): return {'name': n, 'targets': list(t), 'low': lo, 'high': hi}

BROAD = (0.7, 2.3); NBROAD = (-2.3, -0.7); ENT = (0.6, 1.5)

CAND = {}

# A2 interleaved cycle: every entangler separated by non-commuting locals on shared qubits
CAND['interleaved_cycle'] = T('Interleaved entangler cycle', [
    g('ry',[0],*BROAD), g('rx',[1],*BROAD),
    g('rzz',[0,1],*ENT),
    g('ry',[1],*BROAD), g('rz',[2],*BROAD),
    g('rxx',[1,2],*ENT),
    g('rx',[2],*BROAD), g('ry',[3],*BROAD),
    g('ryy',[2,3],*ENT),
    g('rz',[3],*BROAD), g('rx',[0],*NBROAD),
    g('rzz',[0,3],*ENT),
])

# A3 repeated pair sandwiching non-commuting locals (deep parameter correlation)
CAND['repeated_pair_sandwich'] = T('Repeated pair sandwich', [
    g('rx',[0],*BROAD), g('ry',[1],*BROAD),
    g('rzz',[0,1],*ENT),
    g('ry',[0],*BROAD), g('rx',[1],*BROAD),
    g('rzz',[0,1],*ENT),
    g('rz',[0],*BROAD), g('ry',[1],*NBROAD),
    g('rxx',[2,3],*ENT),
    g('rz',[2],*BROAD), g('rx',[3],*BROAD),
    g('ryy',[2,3],*ENT),
])

# A9 nested commutator generating weight-3 terms outside the insertion dictionary
CAND['weight3_nest'] = T('Nested weight-3 generator', [
    g('rzz',[0,1],*ENT), g('rxx',[1,2],*ENT), g('rzz',[0,1],(-1.5),(-0.6)), g('rxx',[1,2],(-1.5),(-0.6)),
    g('ry',[0],*BROAD), g('rz',[1],*BROAD), g('rx',[2],*BROAD), g('ry',[3],*BROAD),
    g('rz',[0],*NBROAD), g('rx',[1],*NBROAD), g('ry',[2],*NBROAD), g('rz',[3],*NBROAD),
])

# A4 star topology, all four entanglers touching q0
CAND['star'] = T('Star of entanglers on q0', [
    g('rx',[0],*BROAD), g('ry',[1],*BROAD), g('rz',[2],*BROAD), g('rx',[3],*BROAD),
    g('rzz',[0,1],*ENT), g('ry',[0],*BROAD),
    g('rxx',[0,2],*ENT), g('rz',[0],*BROAD),
    g('ryy',[0,3],*ENT), g('rx',[0],*BROAD),
    g('rzz',[0,1],*ENT), g('ry',[2],*NBROAD),
])

# A5 near-Clifford frame with broad entanglers (decoy: looks like a Clifford frame)
CAND['near_clifford'] = T('Near-quarter-turn frame', [
    g('rx',[0],1.45,1.69), g('ry',[1],1.45,1.69), g('rz',[2],1.45,1.69), g('rx',[3],1.45,1.69),
    g('rzz',[0,1],1.45,1.69), g('rxx',[2,3],1.45,1.69),
    g('ry',[0],0.8,1.3), g('rz',[1],0.8,1.3), g('rx',[2],0.8,1.3), g('ry',[3],0.8,1.3),
    g('ryy',[1,2],1.45,1.69), g('rzz',[0,3],1.45,1.69),
])

# A7 broad everything: maximal-entropy scrambler
CAND['scrambler'] = T('Broad-angle scrambler', [
    g('ry',[0],-3.1,3.1), g('rx',[1],-3.1,3.1), g('rz',[2],-3.1,3.1), g('ry',[3],-3.1,3.1),
    g('rzz',[0,1],-3.1,3.1), g('rxx',[1,2],-3.1,3.1),
    g('rz',[0],-3.1,3.1), g('ry',[1],-3.1,3.1), g('rx',[2],-3.1,3.1), g('rz',[3],-3.1,3.1),
    g('ryy',[2,3],-3.1,3.1), g('rxx',[0,3],-3.1,3.1),
])

# A8 shared parameter across non-adjacent gates (correlated, non-cancelling)
CAND['shared_chain'] = T('Shared-parameter chain', [
    {'name':'rzz','targets':[0,1],'parameter':'a','scale':1},
    {'name':'ry','targets':[1],'parameter':'b','scale':1},
    {'name':'rxx','targets':[1,2],'parameter':'a','scale':1},
    {'name':'rz','targets':[2],'parameter':'b','scale':-1},
    {'name':'ryy','targets':[2,3],'parameter':'a','scale':-1},
    {'name':'rx','targets':[3],'parameter':'b','scale':1},
    {'name':'rzz','targets':[0,3],'parameter':'a','scale':1},
    {'name':'ry','targets':[0],'parameter':'b','scale':-1},
    g('rx',[0],*BROAD), g('rz',[1],*BROAD), g('ry',[2],*BROAD), g('rx',[3],*NBROAD),
], parameters={'a':{'low':0.8,'high':1.4},'b':{'low':0.9,'high':1.6}})

# reference
CAND['public_ladder'] = json.load(open('/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev/attacks/4q_ladder.json'))

SEEDS = [41, 73, 907]
print(f"{'archetype':<24}{'seed':>6}{'pts':>8}{'cp1':>8}{'cp2':>8}{'cp3':>8}{'eps_final':>12}")
print("-"*74)
summary={}
for name, payload in CAND.items():
    tpl = Template.model_validate(payload)
    try:
        rd = sampling_readiness(tpl, R)
    except ValueError as e:
        print(f"{name:<24} READINESS FAIL: {str(e)[:60]}"); continue
    pts=[]
    for s in SEEDS:
        circ = instantiate(tpl, s, R)
        case = AttackCase(name, circ, s, 'template', name)
        row = run_case(case, R, baseline_defender)
        cp = row['checkpoint_points']+[float('nan')]*3
        print(f"{name:<24}{s:>6}{row['recovery_points']:>8.1f}{cp[0]:>8.1f}{cp[1]:>8.1f}{cp[2]:>8.1f}{row['final_infidelity']:>12.3e}")
        pts.append(row['recovery_points'])
    summary[name]=(float(np.mean(pts)), float(np.max(pts)), rd['qualifying'])
print()
print(f"{'archetype':<24}{'mean_recovery':>15}{'worst_case_for_atk':>20}{'readiness':>11}")
for k,(m,mx,rd) in sorted(summary.items(), key=lambda kv: kv[1][0]):
    print(f"{k:<24}{m:>15.1f}{mx:>20.1f}{rd:>11}")
