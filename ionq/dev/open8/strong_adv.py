"""Overpowered offline adversary: all 96k shots at once, 720 random settings, long pursuit."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import pursuit
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
which, draw, budget = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
t = json.load(open('quantum_duel_work/quantum-duel-8q-open-0.7.1/attacks.json'))[which]
circ = instantiate(Template.model_validate(t), draw, lab.RULES)
d = simulate(circ, shots=96000, S=720, seed=draw)
P = pursuit.Pursuit(8, max_gates=108, max_ents=36, top=20)
t0 = time.process_time(); tr = []
arch, ang, v = P.run(d, time.process_time() + budget, trace=tr)
e = eps_model(arch, ang, circ)
print(json.dumps(dict(attack=t['name'], draw=draw, budget=budget, found=len(arch), eps=e, pts=points(e), cpu=round(time.process_time()-t0))), flush=True)
