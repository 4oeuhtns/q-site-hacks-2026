"""Port the measured frame learner to the real receipt-only SDK contract.
No generator, organizer files, private labels, or judge scores are imported.
"""
import numpy as np
from .recovery import Recovery
from duelkit.quantum import G,validate,experiment_index,bitstrings

def run_defender(client,rules):
    if rules.qubits!=8:raise ValueError('Frame defender expects the eight-qubit profile')
    class Bridge:
        def __init__(self):self.records=[];self.stage=0;self.ceiling=0;self.remaining=0;self.settings_remaining=0;self.counter=0
        def query(self,prep,basis,shots):
            self.counter+=1
            # Research labels are already product eigenstates, q0 leftmost.
            basis=''.join(basis);idx=experiment_index(prep,basis,rules.qubits)
            r=client.query(idx,int(shots),f'frame-{self.stage}-{self.counter}')
            self.remaining-=int(shots)
            self.seen.add(idx);self.settings_remaining=rules.max_settings-len(self.seen)
            c=[r['counts'][b] for b in bitstrings(rules.qubits)]
            self.records.append(dict(prep=list(prep),basis=list(basis),counts=c,shots=int(shots),stage=self.stage))
            return np.array(c,dtype=np.int64)
    bridge=Bridge();bridge.seen=set();learner=Recovery(8,rules.patch_max_gates,rules.patch_max_entanglers,seed=917)
    last=()
    for stage in range(1,rules.checkpoints+1):
        state=client.status();bridge.stage=stage;bridge.ceiling=stage*rules.block
        bridge.remaining=state['available_now'];bridge.settings_remaining=rules.max_settings-state['distinct_settings']
        proposed,meta=learner.checkpoint(bridge)
        converted=[]
        for g in proposed:
            active=[i for i,a in enumerate(g['pauli']) if a!='I'];axes=[g['pauli'][i].lower() for i in active]
            if not 1<=len(active)<=2 or len(set(axes))!=1:raise ValueError('Compiler returned an illegal physical rotation')
            converted.append(G('r'+''.join(axes),tuple(active),float(g['angle'])))
        last=tuple(converted);validate(last,**rules.validation_kwargs())
        client.submit_patch(last,note='Count-only frame recovery; '+meta.get('status','checkpoint'))
        client.close_checkpoint()
