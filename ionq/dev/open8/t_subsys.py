import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, json
sys.path.insert(0,'dev/open8')
from offline import *
import panel_diag, subsys
for spec, draw in [(dict(family="generic",seed=41,bands=[[0.6,1.2],[1.9,2.5]],qubits=[0,1,2]),1),
                   (dict(family="generic",seed=42,bands=[[0.6,1.2],[1.9,2.5]],qubits=[0,1,2]),2),
                   (dict(family="generic",seed=41,bands=[[0.6,1.2],[1.9,2.5]],qubits=[0,1,2,3]),1),
                   (dict(family="pub",name="multilayer"),1)]:
    circ=get_circ(spec,draw); recs=panel_diag.f2def_records(circ,draw)
    sup,seen=subsys.support(recs,8)
    true_sup=sorted({q for g in circ for q in g.targets})
    line=dict(attack=lab.spec_to_template(spec)['name'], true_support=true_sup, detected=sup)
    if 1<=len(sup)<=3:
        t=time.process_time(); arch,x,v=subsys.fit(recs,sup,8,time.process_time()+120)
        e=eps_model(arch,x,circ); line.update(gates=len(arch),eps=f"{e:.2e}",pts=round(points(e),1),cpu=round(time.process_time()-t))
    print(json.dumps(line),flush=True)
