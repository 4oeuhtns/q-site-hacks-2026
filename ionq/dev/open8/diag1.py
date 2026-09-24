import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMBA_NUM_THREADS"): os.environ.setdefault(_v,"1")
import sys, time, collections
sys.path.insert(0,'dev/open8'); import lab
from offline import *
import opendef, f2def
from qduel_sdk.local import LocalSession
spec=dict(family="generic",seed=22,gates=24,ents=8,bands=[[0.6,1.2],[1.9,2.5]])
circ=get_circ(spec,2)
sess=LocalSession(circ,lab.RULES,seed=2+100000)
# wrap time per source
for name in ['run']:
    pass
orig_t=opendef.TemplateSource.run; orig_p=opendef.PursuitSource.run; orig_f=f2def.Frame2Recovery.checkpoint
def timed(f,label):
    def w(*a,**k):
        t=time.process_time(); r=f(*a,**k); print(f'{label}: {time.process_time()-t:.1f}s', file=sys.stderr); return r
    return w
opendef.TemplateSource.run=timed(orig_t,'templates'); opendef.PursuitSource.run=timed(orig_p,'pursuit'); f2def.Frame2Recovery.checkpoint=timed(orig_f,'f2def')
opendef.run_defender(sess.client(), lab.RULES)
sess.client().finish(); res=sess.result()
print('points',[round(c['recovery_points'],1) for c in res['checkpoint_scores']], 'settings',res['distinct_settings'])
recs=[r['payload'] for r in res['records'] if r['kind']=='measurement']
shots=collections.Counter(); bases=collections.Counter(); preps=collections.Counter()
for r in recs:
    bases[r['basis']]+=r['shots']; preps[''.join(p[0] for p in r['prep'])]+=r['shots']
print('distinct bases',len(bases),'top',bases.most_common(5)); print('distinct preps',len(preps),'top',preps.most_common(5))
