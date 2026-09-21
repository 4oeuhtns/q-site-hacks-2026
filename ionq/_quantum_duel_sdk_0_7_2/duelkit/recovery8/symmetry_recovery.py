"""Exploratory count-only estimate from learned surviving Pauli correspondences.
Produces a dense diagnostic estimate; NO claim of a legal compact correction.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .quantum import unitary,inverse,design,measure_change,B,kron,P1
from .clifford import pmul,rank,coefficients,complete_images,best_synthesis,fmt,parse,nullspace

def omega(a,b):return ((a[0]&b[1]).bit_count()+(a[1]&b[0]).bit_count())%2

def pairmul(a,b):
 i=pmul(a[0],b[0]);o=pmul(a[1],b[1]);r=o[2]/i[2]
 if abs(r.imag)>1e-8:raise ValueError('inconsistent paired phase')
 return ((i[0],i[1],1),(o[0],o[1],int(round(r.real))))

def solve(equations,rhs,width):
 rows=[int(a)|(int(b)<<width) for a,b in zip(equations,rhs)];piv=[];r=0
 for c in range(width):
  ix=next((i for i in range(r,len(rows)) if rows[i]>>c&1),None)
  if ix is None:continue
  rows[r],rows[ix]=rows[ix],rows[r]
  for j in range(len(rows)):
   if j!=r and rows[j]>>c&1:rows[j]^=rows[r]
  piv.append(c);r+=1
 for row in rows[r:]:
  if row&((1<<width)-1)==0 and row>>width&1:raise ValueError('no symplectic extension')
 return sum((1<<c) for row,c in zip(rows,piv) if row>>width&1)

def partner(vectors,rhs,n):
 cs=[z|(x<<n) for x,z,_ in vectors];a=solve(cs,rhs,2*n)
 return (a&((1<<n)-1),a>>n,1)

def extend(pairs,n):
 work=list(pairs);basis=[];aa=[];bb=[]
 while work:
  idx=next(((i,j) for i in range(len(work)) for j in range(i+1,len(work)) if omega(work[i][0],work[j][0])),None)
  if idx is None:break
  i,j=idx;a,b=work[i],work[j];remaining=[w for k,w in enumerate(work) if k not in (i,j)];new=[]
  for w in remaining:
   wa,wb=omega(w[0],a[0]),omega(w[0],b[0])
   if wb:w=pairmul(w,a)
   if wa:w=pairmul(w,b)
   new.append(w)
  aa.append(a);bb.append(b);work=new
 # The remaining rows are radical directions; choose matching partners independently.
 for i,zpair in enumerate(work):
  existing=[w for ab in zip(aa,bb) for w in ab];constraints=existing+work
  rhs=[0]*len(existing)+[int(j==i) for j in range(len(work))]
  xp=partner([w[0] for w in constraints],rhs,n);xq=partner([w[1] for w in constraints],rhs,n)
  aa.append((xp,xq));bb.append(zpair)
 # Complete the unconstrained symplectic complement.
 while len(aa)<n:
  existing=[w for ab in zip(aa,bb) for w in ab];free=[]
  for side in [0,1]:
   for bit in range(2*n):
    x=(1<<bit)&((1<<n)-1);z=(1<<bit)>>n;v=(x,z,1)
    for a,b in zip(aa,bb):
     va,vb=omega(v,a[side]),omega(v,b[side])
     if vb:
      p=pmul(v,a[side]);v=(p[0],p[1],1)
     if va:
      p=pmul(v,b[side]);v=(p[0],p[1],1)
    if v[0] or v[1]:break
   free.append(v)
  a=tuple(free);constraints=existing+[a];rhs=[0]*len(existing)+[1]
  b=(partner([w[0] for w in constraints],rhs,n),partner([w[1] for w in constraints],rhs,n))
  aa.append(a);bb.append(b)
 return aa+bb

class SymmetryFit:
 def __init__(self,n,pairs):
  self.n=n
  rows=complete_images(extend(pairs,n),n)
  # Synthesis is used only to construct a consistent Clifford estimate; no cap waiver is hidden.
  self.clifford_inverse=best_synthesis(rows,n,trials=8,gate_cap=1000,ent_cap=1000)
  self.C0=unitary(inverse(self.clifford_inverse),n)
  equations=[z|(x<<n) for (x,z,_),_ in pairs]
  ker=nullspace(equations,2*n)
  self.kernel_dimension=len(ker)
  if len(ker)>4:raise ValueError('exploratory residual model limited to <=15 coefficients')
  vals=[]
  for mask in range(1,1<<len(ker)):
   val=0
   for j,k in enumerate(ker):
    if mask>>j&1:val^=k
   vals.append(val)
  self.labels=[fmt(v&((1<<n)-1),v>>n,n) for v in vals]
  self.ps=np.array([kron([P1[a] for a in label]) for label in self.labels])
 def matrix(self,x):
  h=np.einsum('k,kij->ij',x,self.ps);vals,v=np.linalg.eigh(h)
  e=np.exp(-.5j*vals);u=(v*e)@v.conj().T
  return u,vals,v
 def fit(self,settings,counts,starts=3,maxiter=140):
  k,b=design(settings);ct=np.asarray(counts,float);N=ct.sum()
  def fun(x):
   V,vals,E=self.matrix(x);U=self.C0@V
   psi=k@U.T;amp=measure_change(psi,b,B);p=abs(amp)**2+1e-12
   loss=-np.sum(ct*np.log(p))/N
   lam=measure_change(-2*ct/p*amp/N,b,B,True);G=lam.T@k.conj();Gv=self.C0.conj().T@G;Ge=E.conj().T@Gv@E
   delta=vals[:,None]-vals[None,:]
   F=-.5j*np.exp(-.25j*(vals[:,None]+vals[None,:]))*np.sinc(delta/(4*np.pi))
   Gh=E@(Ge*F.conj())@E.conj().T
   grad=np.einsum('kij,ij->k',self.ps.conj(),Gh).real
   return float(loss),grad
  rng=np.random.default_rng(9891);fits=[]
  for s in range(starts):
   x=np.zeros(len(self.ps)) if s==0 else rng.uniform(-1.5,1.5,len(self.ps))
   fit=minimize(fun,x,jac=True,method='L-BFGS-B',bounds=[(-2*np.pi,2*np.pi)]*len(x),options={'maxiter':maxiter,'maxfun':220,'gtol':1e-7,'ftol':1e-10})
   fits.append(fit)
  f=min(fits,key=lambda f:f.fun);U=self.C0@self.matrix(f.x)[0]
  return U.conj().T,{'n_parameters':len(self.ps),'kernel_dimension':self.kernel_dimension,'loss':float(f.fun),'fit_success':bool(f.success),'nfev_total':sum(f.nfev for f in fits),'coefficients':f.x.tolist(),'residual_generators':self.labels,'clifford_inverse':self.clifford_inverse}
