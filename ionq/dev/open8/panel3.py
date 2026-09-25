"""Does f2def's adaptive panel hurt stage-3 learning? Compare panels at 96k shots."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import pursuit, f2def
from qduel_sdk.local import LocalSession
from duelkit.quantum import experiment_index, bitstrings
LABELS=['0','1','+','-','+i','-i']

def records(circ, draw, mode):
    rules=lab.RULES; sess=LocalSession(circ,rules,seed=draw+100000); client=sess.client()
    rng=np.random.default_rng(draw+7)
    class B:
        records=[]; stage=0; ceiling=0; remaining=0; settings_remaining=0; counter=0; seen=set()
        def query(self,prep,basis,shots):
            self.counter+=1; basis=''.join(basis); idx=experiment_index(prep,basis,8)
            r=client.query(idx,int(shots),f'd-{self.stage}-{self.counter}')
            self.remaining-=int(shots); self.seen.add(idx); self.settings_remaining=rules.max_settings-len(self.seen)
            c=[r['counts'][b] for b in bitstrings(8)]
            self.records.append(dict(prep=list(prep),basis=list(basis),counts=c,shots=int(shots),stage=self.stage)); return np.array(c)
    b=B(); b.records=[]; b.seen=set()
    L=f2def.Frame2Recovery(8,rules.patch_max_gates,rules.patch_max_entanglers,seed=917)
    for stage in (1,2,3):
        st=client.status(); b.stage=stage; b.ceiling=stage*rules.block; b.remaining=st['available_now']
        b.settings_remaining=rules.max_settings-st['distinct_settings']
        if stage==1 or mode=='f2def':
            L.checkpoint(b)
        else:  # uniform random panel on the remaining settings budget
            free=rules.max_settings-len(b.seen); S=max(1,free//(4-stage))
            per=b.remaining//S
            for i in range(S):
                prep=[LABELS[j] for j in rng.integers(6,size=8)]; basis=''.join(rng.choice(list('XYZ'),8))
                b.query(prep,basis,per+(1 if i < b.remaining-per*S else 0) if i==S-1 else per)
            if client.status()['available_now']>0:
                b.query(b.records[-1]['prep'],b.records[-1]['basis'],client.status()['available_now'])
        client.submit_patch((),'x'); client.close_checkpoint()
    return b.records

g,seed,draw,mode=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
spec=dict(family="generic",seed=seed,gates=g,ents=g//3,bands=[[0.6,1.2],[1.9,2.5]])
circ=get_circ(spec,draw)
if mode=='random':
    d=simulate(circ,shots=96000,S=720,seed=draw); info={}
else:
    recs=records(circ,draw,mode); d=sim8.records_to_fast(recs,8)
    per=d.counts.sum(1); info=dict(settings=len(per),max_shots=int(per.max()),median=float(np.median(per)))
P=pursuit.Pursuit(8,max_gates=72,max_ents=24)
t=time.process_time(); arch,ang,v=P.run(d,time.process_time()+380)
e=eps_model(arch,ang,circ)
print(json.dumps(dict(g=g,seed=seed,draw=draw,mode=mode,found=len(arch),eps=round(e,5),pts=round(points(e),1),cpu=round(time.process_time()-t),**info)),flush=True)
