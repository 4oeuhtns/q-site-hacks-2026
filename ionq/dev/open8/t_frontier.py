import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import pursuit
algo, g, seed, draw, shots, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
spec=dict(family="generic",seed=seed,gates=g,ents=g//3,bands=[[0.6,1.2],[1.9,2.5]])
circ=get_circ(spec,draw)
d=simulate(circ,shots=shots,S=360 if shots<=32000 else 720,seed=draw)
P = pursuit.Pursuit(8,max_gates=108,max_ents=36) if algo=='p1' else pursuit.Pursuit2(8,max_gates=108,max_ents=36)
t=time.process_time(); arch,ang,v=P.run(d,time.process_time()+budget)
e=eps_model(arch,ang,circ)
print(json.dumps(dict(algo=algo,g=g,seed=seed,draw=draw,shots=shots,found=len(arch),eps=round(e,5),pts=round(points(e),1),cpu=round(time.process_time()-t),stats=getattr(P,'stats',None))),flush=True)
