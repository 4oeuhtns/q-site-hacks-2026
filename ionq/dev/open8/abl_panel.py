import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import pursuit
mode, seed, draw = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
g = int(sys.argv[4]) if len(sys.argv) > 4 else 24
spec=dict(family="generic",seed=seed,gates=g,ents=g//3,bands=[[0.6,1.2],[1.9,2.5]])
circ=get_circ(spec,draw)
S = {'r360':360,'r720sub160':720,'r720':720,'r160':160,'r180':180}[mode]
d=simulate(circ,S=S,seed=draw)
if mode=='r720sub160':
    d=d.subset(np.argsort(-d.counts.sum(1))[:160])
P=pursuit.Pursuit(8,max_gates=60)
t=time.process_time(); arch,ang,v=P.run(d,time.process_time()+300)
e=eps_model(arch,ang,circ)
print(json.dumps(dict(mode=mode,seed=seed,draw=draw,g=g,found=len(arch),eps=e,pts=points(e),cpu=round(time.process_time()-t))),flush=True)
