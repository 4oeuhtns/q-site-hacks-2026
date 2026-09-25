"""How many settings/shots does f2def need at checkpoint 1 on frames?"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import f2def
from qduel_sdk.local import LocalSession
from duelkit.quantum import experiment_index, bitstrings, G, unitary, infidelity
spec=json.loads(sys.argv[1]); draw=int(sys.argv[2]); cap_set=int(sys.argv[3]); cap_shots=int(sys.argv[4])
circ=get_circ(spec,draw); rules=lab.RULES
sess=LocalSession(circ,rules,seed=draw+100000); client=sess.client()
class B:
    records=[]; stage=1; ceiling=rules.block; remaining=0; settings_remaining=0; counter=0; seen=set()
    def query(self,prep,basis,shots):
        self.counter+=1; basis=''.join(basis); idx=experiment_index(prep,basis,8)
        r=client.query(idx,int(shots),f'd-{self.counter}'); self.remaining-=int(shots); self.seen.add(idx)
        self.settings_remaining=cap_set-len(self.seen)
        c=[r['counts'][b] for b in bitstrings(8)]
        self.records.append(dict(prep=list(prep),basis=list(basis),counts=c,shots=int(shots),stage=1)); return np.array(c)
b=B(); b.records=[]; b.seen=set(); b.remaining=cap_shots; b.settings_remaining=cap_set
L=f2def.Frame2Recovery(8,rules.patch_max_gates,rules.patch_max_entanglers,seed=917)
t=time.process_time(); patch,meta=L.checkpoint(b)
conv=tuple(G('r'+''.join(g['pauli'][i].lower() for i,a in enumerate(g['pauli']) if a!='I')[:1]*(sum(a!='I' for a in g['pauli'])), tuple(i for i,a in enumerate(g['pauli']) if a!='I'), float(g['angle'])) for g in patch)
e=infidelity(unitary(conv,8)@unitary(circ,8)) if conv else 1.0
print(json.dumps(dict(attack=lab.spec_to_template(spec)['name'][:30],cap_settings=cap_set,cap_shots=cap_shots,used_settings=len(b.seen),used_shots=cap_shots-b.remaining,rank=L.parity.rank,status=meta.get('status'),eps=f'{e:.1e}',pts=round(points(e),1),cpu=round(time.process_time()-t))),flush=True)
