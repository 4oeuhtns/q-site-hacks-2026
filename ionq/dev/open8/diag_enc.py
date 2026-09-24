import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8'); import lab
from offline import *
import opendef
from qduel_sdk.local import LocalSession
spec=json.loads(sys.argv[1]); draw=int(sys.argv[2])
circ=get_circ(spec,draw)
sess=LocalSession(circ,lab.RULES,seed=draw+100000)
t=time.process_time()
opendef.run_defender(sess.client(), lab.RULES)
sess.client().finish(); res=sess.result()
print('POINTS',[round(c['recovery_points'],1) for c in res['checkpoint_scores']],'cpu',round(time.process_time()-t), file=sys.stderr)
