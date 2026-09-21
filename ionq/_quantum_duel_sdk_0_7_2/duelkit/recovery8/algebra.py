"""Count-only symmetry algebra estimation and Pauli factorization.
All model construction uses observed Pauli correspondences. No circuit source is accepted.
"""
import numpy as np
from scipy.optimize import minimize
from .quantum import unitary,inverse,design,measure_change,B,kron,P1,monomial
from .clifford import pmul,parse,fmt,complete_images,best_synthesis,nullspace
from .symmetry_recovery import extend

class Algebra:
 def __init__(self,n,pairs):
  self.n=n
  rows=complete_images(extend(pairs,n),n)
  self.cinv=best_synthesis(rows,n,trials=48,gate_cap=1000,ent_cap=1000)
  self.C=unitary(inverse(self.cinv),n)
  ker=nullspace([z|(x<<n) for (x,z,_),_ in pairs],2*n)
  if len(ker)>4:raise ValueError('residual dimension exceeds frozen 15-parameter model')
  self.k=len(ker);D=1<<self.k; vals=[]
  for mask in range(D):
   val=0
   for j,k in enumerate(ker):
    if mask>>j&1:val^=k
   vals.append(val)
  self.labels=[fmt(v&((1<<n)-1),v>>n,n) for v in vals];self.L=np.zeros((D,D,D),complex)
  index={v:i for i,v in enumerate(vals)}
  for i,a in enumerate(self.labels):
   x,z=parse(a)
   for j,b in enumerate(self.labels):
    u,v=parse(b);w,t,ph=pmul((x,z,1),(u,v,1));self.L[i,index[w|(t<<n)],j]=ph
  self.ps=np.stack([kron([P1[c] for c in s]) for s in self.labels])
 def features(self,settings):
  k,b=design(settings); features=[]
  for label in self.labels:
   perm,phase=monomial(label)
   evolved=(k[:,perm]*phase)@self.C.T
   features.append(measure_change(evolved,b,B).reshape(-1))
  return np.asarray(features).T
 def coefficients(self,x,gradient=False):
  h=np.einsum('j,jab->ab',x,self.L[1:]);vals,V=np.linalg.eigh(h);e=np.exp(-.5j*vals)
  coeff=(V*e)@V[0].conj()
  if not gradient:return coeff
  div=-.5j*np.exp(-.25j*(vals[:,None]+vals[None,:]))*np.sinc((vals[:,None]-vals[None,:])/(4*np.pi))
  dH=np.einsum('ia,kij,jb->kab',V.conj(),self.L[1:],V)
  dc=np.einsum('ai,kij,ij,j->ka',V,dH,div,V[0].conj())
  return coeff,dc
 def fit(self,settings,counts,seed=73,starts=10,xold=None):
  T=self.features(settings);weights=np.asarray(counts,float).ravel();N=weights.sum();last={}
  def fun(x):
   co,dc=self.coefficients(x,True);a=T@co;p=abs(a)**2+1e-12
   loss=-np.dot(weights,np.log(p))/N
   ga=-2*(weights/p)*a/N
   gc=T.conj().T@ga
   grad=np.real(dc.conj()@gc)
   return loss,grad
  rng=np.random.default_rng(seed);fits=[]
  for j in range(starts):
   x=xold if j==0 and xold is not None else np.zeros(len(self.labels)-1) if j==0 else rng.normal(0,[1.,2.5,4.][j%3],len(self.labels)-1)
   f=minimize(fun,x,jac=True,method='L-BFGS-B',options={'maxiter':300,'maxls':30,'ftol':1e-11,'gtol':1e-7})
   fits.append(f)
  f=min(fits,key=lambda t:t.fun);coef=self.coefficients(f.x);res=np.einsum('i,iab->ab',coef,self.L)
  return res,{'x':f.x.tolist(),'loss':float(f.fun),'starts':starts,'nfev':sum(f.nfev for f in fits),'kernel_dim':self.k}

def factor_paulis(target,L,maxg=24,tol=1e-7):
 """Greedy best insertion + joint angle refinement of a learned matrix (not true A)."""
 D=len(target);eye=np.eye(D,dtype=complex);seq=[];xx=np.array([]);best=1.
 def mats(ids,x):return [np.cos(t/2)*eye-1j*np.sin(t/2)*L[i] for i,t in zip(ids,x)]
 def eval(ids,x,deriv=True):
  gs=mats(ids,x);pre=[eye]
  for g in gs:pre.append(g@pre[-1])
  tail=target.conj().T.copy();over=np.trace(tail@pre[-1])/D;gds=[]
  for j in range(len(gs)-1,-1,-1):
   do=np.trace(tail@(-.5j*L[ids[j]]@pre[j+1]))/D;gds.append(-2*np.real(over.conjugate()*do));tail=tail@gs[j]
  return float(1-abs(over)**2),np.array(gds[::-1])
 for step in range(maxg):
  gs=mats(seq,xx);pre=[eye]
  for g in gs:pre.append(g@pre[-1])
  posts=[eye]*(len(seq)+1);post=eye
  for pos in range(len(seq)-1,-1,-1):post=post@gs[pos];posts[pos]=post
  opts=[]
  for pos in range(len(seq)+1):
   U=pre[pos]@target.conj().T@posts[pos];a=np.trace(U)/D
   for i in range(1,len(L)):
    bb=-1j*np.trace(L[i]@U)/D
    H=np.array([[abs(a)**2,np.real(a.conjugate()*bb)],[np.real(a.conjugate()*bb),abs(bb)**2]])
    val,v=np.linalg.eigh(H);theta=2*np.arctan2(v[1,-1],v[0,-1]);theta=(theta+np.pi)%(2*np.pi)-np.pi
    opts.append((1-val[-1],pos,i,theta))
  candidates=[]
  seen=set()
  for _,pos,i,ang in sorted(opts):
   ids=seq[:pos]+[i]+seq[pos:];key=tuple(ids)
   if key in seen:continue
   seen.add(key);x=np.insert(xx,pos,ang)
   f=minimize(lambda x:eval(ids,x),x,jac=True,method='L-BFGS-B',options={'maxiter':140,'ftol':1e-12,'gtol':1e-9})
   candidates.append((f.fun,ids,f.x))
   if len(candidates)>=4:break
  best,seq,xx=min(candidates,key=lambda z:z[0])
  if best<tol:break
 return seq,xx,float(best)
