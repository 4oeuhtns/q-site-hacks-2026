"""P0: is f2def's measurement panel the reason in-defender pursuit fails?"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json, pickle
sys.path.insert(0,'dev/open8')
from offline import *
import pursuit, f2def
from qduel_sdk.local import LocalSession
from duelkit.quantum import experiment_index, bitstrings

def f2def_records(circ, draw):
    rules = lab.RULES
    sess = LocalSession(circ, rules, seed=draw + 100000); client = sess.client()
    class Bridge:
        records=[]; stage=0; ceiling=0; remaining=0; settings_remaining=0; counter=0; seen=set()
        def query(self, prep, basis, shots):
            self.counter += 1; basis=''.join(basis)
            idx = experiment_index(prep, basis, 8)
            r = client.query(idx, int(shots), f'd-{self.stage}-{self.counter}')
            self.remaining -= int(shots); self.seen.add(idx)
            self.settings_remaining = rules.max_settings - len(self.seen)
            c = [r['counts'][b] for b in bitstrings(8)]
            self.records.append(dict(prep=list(prep), basis=list(basis), counts=c, shots=int(shots), stage=self.stage))
            return np.array(c, dtype=np.int64)
    b = Bridge(); b.records = []; b.seen = set()
    L = f2def.Frame2Recovery(8, rules.patch_max_gates, rules.patch_max_entanglers, seed=917)
    st = client.status(); b.stage = 1; b.ceiling = rules.block; b.remaining = st['available_now']
    b.settings_remaining = rules.max_settings - st['distinct_settings']
    L.checkpoint(b)
    return b.records

mode, seed, draw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
spec = dict(family="generic", seed=seed, gates=24, ents=8, bands=[[0.6,1.2],[1.9,2.5]])
circ = get_circ(spec, draw)
if mode == 'f2def':
    recs = f2def_records(circ, draw)
    d = sim8.records_to_fast(recs, 8)
    per = d.counts.sum(1)
    info = dict(settings=len(per), shots=int(per.sum()), max_shots=int(per.max()), median=float(np.median(per)))
else:
    d = simulate(circ, shots=32000, S=int(mode[1:]), seed=draw); info = {}
P = pursuit.Pursuit(8, max_gates=60)
t = time.process_time(); arch, ang, v = P.run(d, time.process_time() + 150)
e = eps_model(arch, ang, circ)
print(json.dumps(dict(mode=mode, seed=seed, draw=draw, found=len(arch), eps=round(e,5), pts=round(points(e),1), cpu=round(time.process_time()-t), **info)), flush=True)
