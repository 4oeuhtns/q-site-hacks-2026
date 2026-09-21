"""Port the recovery8 FRAME-AWARE learner to 4 qubits and measure it.

recovery8/quantum.py is dimension-general and Recovery takes n as a parameter;
the only hard 8 is the adapter's guard. This mirrors recovery8/adapter.py with
n = rules.qubits so the real algebraic frame learner runs on Cup 1.
"""
import sys, time, json
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev")
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from duelkit.quantum import G, validate, experiment_index, bitstrings
from duelkit.recovery8.recovery import Recovery
from qduel_sdk.rules import Rules
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from harness import run_case, AttackCase, baseline_defender, load_attack, load_defender

def frame_defender(client, rules):
    n = rules.qubits
    class Bridge:
        def __init__(self):
            self.records=[]; self.stage=0; self.ceiling=0; self.remaining=0
            self.settings_remaining=0; self.counter=0; self.seen=set()
        def query(self, prep, basis, shots):
            self.counter += 1
            basis = ''.join(basis)
            idx = experiment_index(prep, basis, n)
            shots = int(min(shots, self.remaining))
            if shots <= 0: raise ValueError('frame learner exhausted its block')
            r = client.query(idx, shots, f'frame-{self.stage}-{self.counter}')
            self.remaining -= shots
            self.seen.add(idx); self.settings_remaining = rules.max_settings - len(self.seen)
            c = [r['counts'][b] for b in bitstrings(n)]
            self.records.append(dict(prep=list(prep), basis=list(basis), counts=c,
                                     shots=shots, stage=self.stage))
            return np.array(c, dtype=np.int64)
    bridge = Bridge()
    learner = Recovery(n, rules.patch_max_gates, rules.patch_max_entanglers, seed=917)
    last = ()
    for stage in range(1, rules.checkpoints+1):
        state = client.status(); bridge.stage = stage; bridge.ceiling = stage*rules.block
        bridge.remaining = state['available_now']
        bridge.settings_remaining = rules.max_settings - state['distinct_settings']
        try:
            proposed, meta = learner.checkpoint(bridge)
        except BaseException as exc:
            proposed, meta = [], {'status': f'{type(exc).__name__}: {str(exc)[:60]}'}
        converted = []
        for g in proposed:
            active = [i for i,a in enumerate(g['pauli']) if a != 'I']
            axes = [g['pauli'][i].lower() for i in active]
            if not 1 <= len(active) <= 2 or len(set(axes)) != 1: continue
            converted.append(G('r'+''.join(axes), tuple(active), float(g['angle'])))
        try:
            validate(tuple(converted), **rules.validation_kwargs())
            last = tuple(converted)
        except (ValueError, TypeError):
            pass
        if last:
            try: client.submit_patch(last, note='4q frame port; '+str(meta.get('status',''))[:60])
            except (ValueError, TypeError): pass
        client.close_checkpoint()

R = Rules()
own = json.load(open('/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/attacks.json'))
mine = load_defender('/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/quantum_duel_work/quantum-duel-4q-playtest-0.3/my_solution/main.py')

cases = []
for t in own:
    tpl = Template.model_validate(t)
    for s in (41, 73, 907):
        cases.append(AttackCase(t['name'][:26], instantiate(tpl,s,R), s, 'template', t['name'][:26]))
for nm in ('4q_ladder','4q_mixed','4q_conjugated','4q_zz'):
    cases += load_attack(f'/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev/attacks/{nm}.json',
                         seeds=(41,), rules=R)

print(f"{'attack':<34}{'stock':>7}{'mine':>7}{'frame4':>8}{'secs':>7}  frame status")
print("-"*82)
for c in cases:
    a = run_case(c, R, baseline_defender)
    b = run_case(c, R, mine)
    t0 = time.time(); f = run_case(c, R, frame_defender); dt = time.time()-t0
    note = f['status'] if f['status']!='PASSED' else ''
    print(f"{c.label[:33]:<34}{a['recovery_points']:>7.1f}{b['recovery_points']:>7.1f}"
          f"{f['recovery_points']:>8.1f}{dt:>7.1f}  {note}{f['error'][:40]}",flush=True)
