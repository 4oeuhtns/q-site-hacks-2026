import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json, cProfile, pstats, io
sys.path.insert(0,'dev/open8')
from offline import *
import f2def
sys.argv += []
spec=json.loads(sys.argv[1]); draw=int(sys.argv[2])
import panel_diag  # reuse the f2def-only bridge for stage 1
circ=get_circ(spec,draw)
pr=cProfile.Profile(); t=time.process_time(); pr.enable()
recs=panel_diag.f2def_records(circ,draw)
pr.disable(); print('stage-1 f2def cpu', round(time.process_time()-t,1))
s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumulative').print_stats(18); print(s.getvalue()[:3500])
