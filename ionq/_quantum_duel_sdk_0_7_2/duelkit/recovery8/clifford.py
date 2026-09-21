"""Count-only Clifford learner using product preparations and readout.
No Bell measurements, ancillas, controlled U, U-dagger, or family labels.
Reconstructs Pauli correspondences from affine supports of bitstring distributions.
A deliberately general synthesis is checked against physical rotation gate caps.
"""
from __future__ import annotations
import numpy as np
from .quantum import canonical_angle

def parse(s):
 n=len(s);x=z=0
 for q,c in enumerate(s):
  if c in 'XY':x|=1<<(n-q-1)
  if c in 'YZ':z|=1<<(n-q-1)
 return x,z

def fmt(x,z,n):return ''.join('I' if not(x&(1<<(n-q-1)) or z&(1<<(n-q-1))) else ('Y' if x&z&(1<<(n-q-1)) else ('X' if x&(1<<(n-q-1)) else 'Z')) for q in range(n))
def pmul(a,b):
 x,z,r=a;u,v,s=b
 ph=(1j)**((x&z).bit_count()+(u&v).bit_count()-((x^u)&(z^v)).bit_count())*((-1)**((z&u).bit_count()))*r*s
 return x^u,z^v,ph

def conjugate(g,row):
 p,q=parse(g['pauli']);x,z,r=row
 if ((p&z).bit_count()+(q&x).bit_count())%2==0:return row
 k=round(float(g['angle'])/(np.pi/2))%4
 if k==0:return row
 if k==2:return x,z,-r
 a,b,c=pmul((p,q,1),(x,z,r));c*=(-1j if k==1 else 1j)
 if abs(c.imag)>1e-8:raise ValueError('phase bug')
 return a,b,int(round(c.real))

def images(gates,n):
 rows=[(1<<(n-q-1),0,1) for q in range(n)]+[(0,1<<(n-q-1),1) for q in range(n)]
 for g in gates:rows=[conjugate(g,r) for r in rows]
 return rows

def rank(rows):
 bas={}
 for a in rows:
  while a:
   k=a.bit_length()-1
   if k not in bas:bas[k]=a;break
   a^=bas[k]
 return len(bas)

def nullspace(rows,n):
 # Brute force is tiny: n<=8, used only to recover parity constraints.
 vals=[v for v in range(1,1<<n) if all((v&r).bit_count()%2==0 for r in rows)]
 out=[];r=0
 for v in vals:
  nr=rank(out+[v])
  if nr>r:out.append(v);r=nr
 return out

def coefficients(rows,w):
 bas={}
 for i,a in enumerate(rows):
  c=1<<i
  while a:
   k=a.bit_length()-1
   if k not in bas:bas[k]=(a,c);break
   p,cc=bas[k];a^=p;c^=cc
 c=0
 while w:
  k=w.bit_length()-1
  if k not in bas:raise ValueError('incomplete basis')
  p,cc=bas[k];w^=p;c^=cc
 return c

def complete_images(pairs,n):
 keys=[x|(z<<n) for (x,z,_),_ in pairs];out=[]
 for q in range(2*n):
  mask=(1<<(n-q-1)) if q<n else (1<<(2*n-(q-n)-1))
  cs=coefficients(keys,mask);inp=(0,0,1);oup=(0,0,1)
  for j,(a,b) in enumerate(pairs):
   if cs>>j&1:inp=pmul(inp,a);oup=pmul(oup,b)
  phase=oup[2]/inp[2]
  if abs(phase.imag)>1e-8:raise ValueError('non-Hermitian mapping')
  out.append((oup[0],oup[1],int(round(phase.real))))
 # Reject inconsistent symplectic data, never replace it using organizer knowledge.
 for i,a in enumerate(out):
  for j,b in enumerate(out):
   antic=((a[0]&b[1]).bit_count()+(a[1]&b[0]).bit_count())%2
   if antic!=int((i<n and j==i+n)or(j<n and i==j+n)):raise ValueError('not symplectic')
 return out

def synthesize_inverse(rows,n,order=None):
 """Clear a learned tableau with legal rotations (not optimal gate minimization)."""
 rows=list(rows);out=[]
 def add(label,theta):
  nonlocal rows
  g={'pauli':label,'angle':canonical_angle(theta)};out.append(g);rows=[conjugate(g,r) for r in rows]
 def local(q,a,t):add('I'*q+a+'I'*(n-q-1),t)
 def cx(c,t):
  local(t,'Y',-np.pi/2);local(c,'Z',np.pi/2);local(t,'Z',np.pi/2)
  add(''.join('Z' if q in (c,t) else 'I' for q in range(n)),-np.pi/2);local(t,'Y',np.pi/2)
 def get(i,q):return fmt(rows[i][0],rows[i][1],n)[q]
 order=list(range(n)) if order is None else list(order)
 remaining=set(order)
 for i in order:
  active=sorted(remaining)
  pivot=i if get(n+i,i)!='I' else next(q for q in active if get(n+i,q)!='I')
  if pivot!=i:cx(i,pivot);cx(pivot,i);cx(i,pivot)
  for q in active:
   c=get(n+i,q)
   if c=='X':local(q,'Y',-np.pi/2)
   elif c=='Y':local(q,'X',np.pi/2)
  for q in active:
   if q==i:continue
   if get(n+i,q)=='Z':cx(q,i)
  if get(i,i)=='Y':local(i,'Z',-np.pi/2)
  for q in active:
   if q==i:continue
   c=get(i,q)
   if c=='Z':local(q,'Y',np.pi/2)
   elif c=='Y':local(q,'Z',-np.pi/2)
   if get(i,q)=='X':cx(i,q)
  if rows[i][2]<0:local(i,'Z',np.pi)
  if rows[n+i][2]<0:local(i,'X',np.pi)
  remaining.remove(i)
 assert rows==images([],n)
 return compress_locals(simplify(out,n),n)

def simplify(gs,n):
 # Combine rotations of the same generator across commuting intervening gates.
 out=[]
 for g in gs:
  new=dict(g);x,z=parse(g['pauli']);joined=False
  for j in range(len(out)-1,-1,-1):
   u,v=parse(out[j]['pauli'])
   if out[j]['pauli']==g['pauli']:
    out[j]['angle']=canonical_angle(out[j]['angle']+g['angle']);joined=True
    if abs(out[j]['angle'])<1e-10:out.pop(j)
    break
   if ((x&v).bit_count()+(z&u).bit_count())%2:break
  if not joined and abs(new['angle'])>1e-10:out.append(new)
 return out

class ParityLearner:
 def __init__(self,n,seed=1001,gate_cap=108,ent_cap=36):self.gate_cap=gate_cap;self.ent_cap=ent_cap;self.n=n;self.rng=np.random.default_rng(seed);self.pairs=[];self.rank=0;self.records=[];self.patch=[];self.complete=False
 def run_until(self,client,cap):
  n=self.n
  while not self.complete and client.remaining>=64*(n+1) and client.settings_remaining>=n+1:
   ai=self.rng.choice(list('XYZ'),n);ao=self.rng.choice(list('XYZ'),n)
   prep=[{'X':'+','Y':'+i','Z':'0'}[a] for a in ai]
   base=client.query(prep,list(ao),64);support=np.flatnonzero(base)
   v0=int(support[0]);parities=nullspace([int(v)^v0 for v in support[1:]],n)
   if not parities:continue
   flipped=[]
   for q in range(n):
    ps=prep.copy();ps[q]={'+':'-','+i':'-i','0':'1'}[ps[q]];flipped.append(client.query(ps,list(ao),64))
   for parity in parities:
    sign=(parity&v0).bit_count()%2;coeff=[];valid=True
    for ct in flipped:
     ss={(int(v)&parity).bit_count()%2 for v in np.flatnonzero(ct)}
     if len(ss)!=1:valid=False;break
     coeff.append(next(iter(ss))^sign)
    if not valid:continue
    inp=''.join(a if bit else 'I' for a,bit in zip(ai,coeff));oup=''.join(ao[q] if parity&(1<<(n-q-1)) else 'I' for q in range(n))
    x,z=parse(inp);u,v=parse(oup);keys=[a[0]|(a[1]<<n) for a,b in self.pairs]
    nr=rank(keys+[x|(z<<n)])
    if nr>self.rank:
     self.pairs.append(((x,z,1),(u,v,(-1)**sign)));self.rank=nr
   if self.rank==2*n:
    try:self.patch=best_synthesis(complete_images(self.pairs,n),n,gate_cap=self.gate_cap,ent_cap=self.ent_cap);self.complete=True
    except ValueError:self.records.append({'inconsistent_full_mapping':True});break
  return self.patch,{'rank':self.rank,'complete':self.complete,'pairs':len(self.pairs)}


def compress_locals(gs,n):
 from .quantum import I,P1
 alphabet=[(a,t) for a in 'XYZ' for t in [-np.pi/2,np.pi/2,np.pi]]
 rots={(a,t):np.cos(t/2)*I-1j*np.sin(t/2)*P1[a] for a,t in alphabet}
 choices=[([],I)]
 choices += [([(a,t)],rots[a,t]) for a,t in alphabet]
 choices += [([a,b],rots[b]@rots[a]) for a in alphabet for b in alphabet]
 pending=[I.copy() for _ in range(n)];out=[]
 def flush(q):
  v=pending[q]
  for seq,u in choices:
   if abs(np.trace(v.conj().T@u))>2-1e-8:
    out.extend({'pauli':'I'*q+a+'I'*(n-q-1),'angle':float(t)} for a,t in seq);pending[q]=I.copy();return
  raise ValueError('local Clifford missing')
 for g in gs:
  support=[q for q,a in enumerate(g['pauli']) if a!='I']
  if len(support)==1:
   q=support[0];a=g['pauli'][q];t=g['angle'];pending[q]=(np.cos(t/2)*I-1j*np.sin(t/2)*P1[a])@pending[q]
  elif abs(abs(g['angle'])-np.pi)<1e-8:
   for q in support:pending[q]=P1[g['pauli'][q]]@pending[q]
  else:
   for q in support:flush(q)
   out.append(g)
 for q in range(n):flush(q)
 return simplify(out,n)

def best_synthesis(rows,n,trials=32,gate_cap=108,ent_cap=36):
 rng=np.random.default_rng(3001);candidates=[]
 for j in range(trials):
  order=list(range(n)) if j==0 else list(rng.permutation(n))
  gs=synthesize_inverse(rows,n,order)
  candidates.append(gs)
 return min(candidates,key=lambda gs:(int(len(gs)>gate_cap or sum(sum(x!='I' for x in g['pauli'])==2 for g in gs)>ent_cap),sum(sum(x!='I' for x in g['pauli'])==2 for g in gs),len(gs)))
