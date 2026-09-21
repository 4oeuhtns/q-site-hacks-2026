"""Dimension-general streamed simulator. q0 = left/MSB, gates chronological, radians.
No full Pauli dictionary, full PTM, or dense Choi matrix is allocated.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import numpy as np
from numba import njit

I=np.eye(2,dtype=complex)
P1={'I':I,'X':np.array([[0,1],[1,0]],complex),'Y':np.array([[0,-1j],[1j,0]],complex),'Z':np.diag([1,-1]).astype(complex)}
KETS={'0':np.array([1,0],complex),'1':np.array([0,1],complex),'+':np.array([1,1],complex)/np.sqrt(2),'-':np.array([1,-1],complex)/np.sqrt(2),'+i':np.array([1,1j],complex)/np.sqrt(2),'-i':np.array([1,-1j],complex)/np.sqrt(2)}
B=np.array([np.column_stack([KETS['+'],KETS['-']]).conj().T,np.column_stack([KETS['+i'],KETS['-i']]).conj().T,I])

def kron(xs):
 r=np.array([1.],complex)
 for x in xs:r=np.kron(r,x)
 return r

def dictionary(n):
 out=[]
 for q in range(n):
  for a in 'XYZ':out.append('I'*q+a+'I'*(n-q-1))
 for p,q in combinations(range(n),2):
  for a in 'XYZ':out.append(''.join(a if i in (p,q) else 'I' for i in range(n)))
 return out

@lru_cache(512)
def monomial(label):
 n=len(label);d=1<<n;x=z=ny=0
 for j,a in enumerate(label):
  bit=1<<(n-1-j)
  if a in 'XY':x|=bit
  if a in 'YZ':z|=bit
  if a=='Y':ny+=1
 perm=np.arange(d,dtype=np.int64)^x
 phase=np.array([(1j)**ny*(-1)**((int(k)&z).bit_count()) for k in perm],complex)
 return perm,phase

def encoded(labels,n):
 if not labels:return np.zeros((0,1<<n),np.int64),np.zeros((0,1<<n),complex)
 m=[monomial(p) for p in labels]
 return np.stack([x[0] for x in m]),np.stack([x[1] for x in m])

@njit(cache=True)
def rot_apply(states,perm,phase,t):
 out=np.empty_like(states);c=np.cos(t/2);s=-1j*np.sin(t/2)
 for k in range(states.shape[0]):
  for j in range(states.shape[1]):out[k,j]=c*states[k,j]+s*phase[j]*states[k,perm[j]]
 return out

@njit(cache=True)
def evolve(states,perms,phases,angles):
 r=states.copy()
 for j in range(len(angles)):r=rot_apply(r,perms[j],phases[j],angles[j])
 return r

@njit(cache=True)
def measure_change(states,bases,b,inverse=False):
 r=states.copy();n=bases.shape[1]
 for q in range(n):
  mask=1<<(n-1-q)
  for s in range(len(r)):
   m=b[bases[s,q]]
   for j in range(r.shape[1]):
    if j&mask:continue
    a,c=r[s,j],r[s,j|mask]
    if inverse:
     r[s,j]=np.conj(m[0,0])*a+np.conj(m[1,0])*c
     r[s,j|mask]=np.conj(m[0,1])*a+np.conj(m[1,1])*c
    else:
     r[s,j]=m[0,0]*a+m[0,1]*c;r[s,j|mask]=m[1,0]*a+m[1,1]*c
 return r

@njit(cache=True)
def objective(angles,perms,phases,kets,bases,counts,b):
 state=evolve(kets,perms,phases,angles)
 amp=measure_change(state,bases,b)
 total=counts.sum();loss=0.;adj=np.empty_like(amp)
 for s in range(amp.shape[0]):
  for j in range(amp.shape[1]):
   p=abs(amp[s,j])**2+1e-12
   loss-=counts[s,j]*np.log(p)/total
   adj[s,j]=-2*counts[s,j]/p*amp[s,j]/total
 adj=measure_change(adj,bases,b,True)
 grads=np.zeros(len(angles))
 for g in range(len(angles)-1,-1,-1):
  value=0.
  for s in range(state.shape[0]):
   for j in range(state.shape[1]):
    value+=(np.conj(adj[s,j])*(-.5j*phases[g,j]*state[s,perms[g,j]])).real
  grads[g]=value
  state=rot_apply(state,perms[g],phases[g],-angles[g]);adj=rot_apply(adj,perms[g],phases[g],-angles[g])
 return loss,grads

@njit(cache=True)
def append_gradients(angles,perms,phases,kets,bases,counts,b,dp,dph):
 state=evolve(kets,perms,phases,angles);amp=measure_change(state,bases,b);total=counts.sum()
 adj=-2*counts/(np.abs(amp)**2+1e-12)*amp/total
 adj=measure_change(adj,bases,b,True)
 grad=np.zeros(len(dp))
 for g in range(len(dp)):
  value=0.
  for s in range(state.shape[0]):
   for j in range(state.shape[1]):value+=(np.conj(adj[s,j])*(-.5j*dph[g,j]*state[s,dp[g,j]])).real
  grad[g]=value
 return grad

def unitary(gates,n):
 labels=[g['pauli'] for g in gates];x=np.array([g['angle'] for g in gates]);p,h=encoded(labels,n)
 return evolve(np.eye(1<<n,dtype=complex),p,h,x).T

def inverse(gates):return [{'pauli':g['pauli'],'angle':-float(g['angle'])} for g in gates[::-1]]
def infidelity(a,p):return float(np.clip(1-abs(np.einsum('ij,ji->',p,a))**2/len(a)**2,0,1))
def old_score(e):return float(np.clip(50*np.log10(.1/max(float(e),1e-16)),0,100))
def wide_score(e):return float(np.clip(100*np.log(.9/max(float(e),1e-16))/np.log(.9/.001),0,100))

def canonical_angle(t):return float((float(t)+np.pi)%(2*np.pi)-np.pi)
def validate(gates,n,gate_cap,ent_cap):
 if len(gates)>gate_cap:raise ValueError('gate cap exceeded')
 ent=0
 for g in gates:
  label=g['pauli'];ts=g['angle'];support=[c for c in label if c!='I']
  if len(label)!=n or any(c not in 'IXYZ' for c in label) or len(support) not in (1,2) or len(set(support))!=1:raise ValueError('illegal generator')
  if not np.isfinite(ts):raise ValueError('nonfinite angle')
  ent+=len(support)==2
 if ent>ent_cap:raise ValueError('entangler cap exceeded')
 return {'gates':len(gates),'entanglers':ent}

def panel(n,m,seed):
 rng=np.random.default_rng(seed);seen=set();out=[]
 while len(out)<m:
  prep=tuple(rng.choice(list(KETS),n));basis=tuple(rng.choice(list('XYZ'),n));key=(prep,basis)
  if key not in seen:seen.add(key);out.append({'prep':list(prep),'basis':list(basis)})
 return out

def design(settings):
 return np.stack([kron([KETS[k] for k in s['prep']]) for s in settings]),np.array([['XYZ'.index(a) for a in s['basis']] for s in settings],np.int64)

def probabilities(gates,n,settings,analysis=()):
 k,m=design(settings);seq=list(gates)+list(analysis);p,h=encoded([g['pauli'] for g in seq],n)
 amp=measure_change(evolve(k,p,h,np.array([g['angle'] for g in seq])),m,B)
 probs=np.abs(amp)**2
 return probs/probs.sum(axis=1,keepdims=True)

def counts_for(prob,shots,seed):
 rng=np.random.default_rng(seed);alloc=np.full(len(prob),shots//len(prob),int);alloc[:shots%len(prob)]+=1
 return np.array([rng.multinomial(int(k),p) for k,p in zip(alloc,prob)],dtype=np.int64)

# Independent dense oracle: exponentiate only 2x2 or 4x4 generators and embed by tensor indexing.
def independent_unitary(gates,n):
 from scipy.linalg import expm
 d=1<<n;u=np.eye(d,dtype=complex)
 for g in gates:
  qs=[i for i,p in enumerate(g['pauli']) if p!='I'];local=kron([P1[g['pauli'][i]] for i in qs])
  r=expm(-.5j*g['angle']*local);rest=[j for j in range(n) if j not in qs]
  axes=qs+rest+[n];t=u.reshape([2]*n+[d]).transpose(axes).reshape(1<<len(qs),-1)
  u=(r@t).reshape([2]*n+[d]).transpose(np.argsort(axes)).reshape(d,d)
 return u

def independent_probs(u,settings):
 out=[]
 for s in settings:
  psi=kron([KETS[x] for x in s['prep']]);basis=kron([B['XYZ'.index(x)] for x in s['basis']]);out.append(abs(basis@(u@psi))**2)
 return np.array(out)
