"""Small commutant Clifford dictionary. Generated from the learned Pauli algebra only."""
import numpy as np
from collections import deque
from .clifford import parse

def clifford_atlas(L):
 D=len(L);k=int(np.log2(D));vs=list(range(1,D));chosen=[]
 # Find symplectic pairs and recurse through their orthogonal complement.
 def anti(i,j):return abs(np.trace(L[i]@L[j]@L[i]@L[j])/D+1)<1e-7
 remaining=vs.copy();zs=[]
 while remaining:
  pair=next(((i,j) for i in remaining for j in remaining if anti(i,j)),None)
  if pair is None:
   chosen+=remaining[:k];break
  i,j=pair;chosen.extend([i,j]);zs.append(j)
  remaining=[a for a in remaining if not anti(a,i) and not anti(a,j)]
 if len(zs)>1:chosen.extend([zs[i]^zs[i+1] for i in range(len(zs)-1)])
 chosen=list(dict.fromkeys(chosen)); eye=np.eye(D,dtype=complex)
 rot={i:(eye-1j*L[i])/np.sqrt(2) for i in chosen}
 def canonical(x):
  nz=np.flatnonzero(abs(x)>1e-7)
  if len(nz):x=x*np.exp(-1j*np.angle(x[nz[0]]))
  x=np.round(x.real,7)+1j*np.round(x.imag,7)
  return x
 v=np.zeros(D,complex);v[0]=1;vectors=[v];words=[[]];seen={np.rint(np.r_[v.real,v.imag]*1e6).astype(np.int64).tobytes()};todo=deque([0])
 while todo:
  ix=todo.popleft()
  for i in chosen:
   # Keep full-precision representatives, use rounded normalized keys for identity.
   c=rot[i]@vectors[ix];can=canonical(c);key=np.rint(np.r_[can.real,can.imag]*1e6).astype(np.int64).tobytes()
   if key not in seen:
    seen.add(key);vectors.append(c);words.append(words[ix]+[i]);todo.append(len(vectors)-1)
    if len(vectors)>50000:raise ValueError('unexpected centralizer Clifford group size')
 return np.array(vectors),words
