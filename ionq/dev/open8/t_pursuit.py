import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import *
import pursuit
specs = json.loads(sys.argv[1]); budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120
for spec in specs:
    circ = get_circ(spec, 1)
    d = simulate(circ, seed=1)
    P = pursuit.Pursuit(8, max_gates=int(spec.get('pg', 40)))
    t0 = time.process_time(); tr = []
    arch, ang, v = P.run(d, time.process_time() + budget, trace=tr)
    e = eps_model(arch, ang, circ)
    print(json.dumps(spec), f"true_gates={len(circ)} found={len(arch)} nll={v:.1f} eps={e:.2e} pts={points(e):.1f} cpu={time.process_time()-t0:.0f}", flush=True)
