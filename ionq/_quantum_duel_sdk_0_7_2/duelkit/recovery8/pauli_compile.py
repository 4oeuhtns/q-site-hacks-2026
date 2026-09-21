"""Synthesize a learned Pauli-rotation network followed by a learned Clifford.
No original attack circuit is used. All final operations are 1/2-qubit same-axis rotations.
"""
import numpy as np
from .clifford import parse,fmt,conjugate,images,best_synthesis,simplify,compress_locals
from .quantum import inverse,canonical_angle

def qgate(n,q,a,t):return {'pauli':'I'*q+a+'I'*(n-q-1),'angle':float(t)}
def row_image(gs,row):
 for g in gs:row=conjugate(g,row)
 return row

def network(paulis,tail_rows,n,seed=0,tail_trials=8):
 """tail_rows is the forward action of the desired trailing Clifford."""
 rng=np.random.default_rng(seed);F=[];prefix=[]
 for orig in paulis:
  p=row_image(F,(*parse(orig['pauli']),1));s=fmt(p[0],p[1],n);supp=[i for i,c in enumerate(s) if c!='I']
  if not supp:continue
  pivot=int(rng.choice(supp));leaves=[q for q in supp if q!=pivot];rng.shuffle(leaves)
  for q in leaves:
   s=fmt(p[0],p[1],n);a=s[q]
   if s[pivot]==a:
    change=str(rng.choice([x for x in 'XYZ' if x!=a]));g=qgate(n,pivot,change,np.pi/2);F.append(g);prefix.append(g);p=conjugate(g,p)
   s=fmt(p[0],p[1],n);g={'pauli':''.join(a if i in (pivot,q) else 'I' for i in range(n)),'angle':np.pi/2}
   F.append(g);prefix.append(g);p=conjugate(g,p)
  s=fmt(p[0],p[1],n);assert sum(c!='I' for c in s)==1
  prefix.append({'pauli':s,'angle':canonical_angle(float(p[2])*orig['angle'])})
 # Need synthesize desired T F^-1. Invert mapping of T F^-1 directly by composing inverse tableau.
 Frows=images(F,n)
 # Input to inverse synthesis is F T^-1. Obtain T^-1 by swapping correspondence basis.
 from .clifford import complete_images
 standard=images([],n)
 Tinverse=complete_images(list(zip(tail_rows,standard)),n)
 target_inv=[]
 for p in Tinverse:target_inv.append(row_image(F,p))
 from .clifford_pair import best_pair
 gs=best_pair(target_inv,n,trials=tail_trials)
 return merge_locals(simplify(prefix+gs,n),n)

def merge_locals(gs,n):
 """Merge arbitrary local rotations using a ZYZ Euler decomposition (up to global phase)."""
 from .quantum import I,P1
 pending=[I.copy() for _ in range(n)];out=[]
 def flush(q):
  u=pending[q];det=np.linalg.det(u);u=u/np.sqrt(det)
  # Prefer short exact local Clifford words where applicable.
  alphabet=[(a,t) for a in 'XYZ' for t in [-np.pi/2,np.pi/2,np.pi]]
  choices=[([],I)]
  rs={(a,t):np.cos(t/2)*I-1j*np.sin(t/2)*P1[a] for a,t in alphabet}
  choices += [([(a,t)],rs[a,t]) for a,t in alphabet]
  choices += [([a,b],rs[b]@rs[a]) for a in alphabet for b in alphabet]
  for seq,v in choices:
   if abs(np.trace(u.conj().T@v))>2-1e-10:
    out.extend(qgate(n,q,a,t) for a,t in seq);pending[q]=I.copy();return
  beta=2*np.arctan2(abs(u[1,0]),abs(u[0,0]))
  if abs(np.sin(beta/2))<1e-10:alpha=-2*np.angle(u[0,0]);gamma=0.
  elif abs(np.cos(beta/2))<1e-10:alpha=2*np.angle(u[1,0]);gamma=0.
  else:alpha=np.angle(u[1,0])-np.angle(u[0,0]);gamma=-np.angle(u[1,0])-np.angle(u[0,0])
  # time order Rz(gamma), Ry(beta), Rz(alpha)
  for a,t in [('Z',gamma),('Y',beta),('Z',alpha)]:
   t=canonical_angle(t)
   if abs(t)>1e-9:out.append(qgate(n,q,a,t))
  pending[q]=I.copy()
 for g in gs:
  qs=[q for q,c in enumerate(g['pauli']) if c!='I'];t=g['angle']
  if len(qs)==1:
   q=qs[0];p=P1[g['pauli'][q]];pending[q]=(np.cos(t/2)*I-1j*np.sin(t/2)*p)@pending[q]
  elif abs(abs(canonical_angle(t))-np.pi)<1e-9:
   for q in qs:pending[q]=P1[g['pauli'][q]]@pending[q]
  else:
   for q in qs:flush(q)
   out.append(g)
 for q in range(n):flush(q)
 return simplify(out,n)

def clifford_absorb(paulis,tail,n,drop=0.):
 """Move quarter-turn pieces to the trailing Clifford; optionally drop tiny residual angles."""
 pending=[];residual=[]
 for g in paulis:
  # pending before this rotation is commuted after it.
  p=row_image(inverse(pending),(*parse(g['pauli']),1));lab=fmt(p[0],p[1],n)
  angle=canonical_angle(g['angle']*float(p[2]));quarter=np.round(angle/(np.pi/2))*(np.pi/2);delta=canonical_angle(angle-quarter)
  if abs(delta)>drop:residual.append({'pauli':lab,'angle':delta})
  if abs(quarter)>1e-10:
   # originally R_g * pending = pending * R_transformed. Pending_new = pending * R_lab(quarter)
   pending=[{'pauli':lab,'angle':float(quarter)}]+pending
 return residual,images(pending+tail,n)

def compile_network(paulis,tail,n,gate_cap=108,ent_cap=36,trials=64,drop=0.):
 non,rows=clifford_absorb(paulis,tail,n,drop)
 best=None
 for s in range(trials):
  gs=network(non,rows,n,s,tail_trials=4)
  ent=sum(sum(c!='I' for c in g['pauli'])==2 for g in gs)
  key=(int(ent>ent_cap or len(gs)>gate_cap),ent,len(gs))
  if best is None or key<best[0]:best=(key,gs)
 return best[1],{'entanglers':best[0][1],'gates':best[0][2],'legal':not best[0][0],'nonclifford_rotations':len(non),'trials':trials,'drop':drop}
