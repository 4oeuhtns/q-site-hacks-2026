"""Count-only learners. Algorithms deliberately do not import oracle/generator modules."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .quantum import *

class DenseMLE:
    """General SU(2**n) MLE with exact spectral Frechet derivative.
    Dense inverse is a diagnostic relaxation: no synthesis penalties are applied.
    """
    def __init__(self,n,maxiter=350,restarts=2):
        self.n=n;self.labels,self.ps=all_paulis(n);self.x=None
        self.maxiter=maxiter;self.restarts=restarts;self.trace=[]
    def matrix(self,x):
        h=np.einsum('k,kij->ij',x,self.ps,optimize=True)/2
        w,v=np.linalg.eigh(h);e=np.exp(-1j*w)
        return (v*e)@v.conj().T
    def objective(self,x,likelihood):
        h=np.einsum('k,kij->ij',x,self.ps,optimize=True)/2
        w,v=np.linalg.eigh(h);e=np.exp(-1j*w);u=(v*e)@v.conj().T
        loss,g=likelihood(u)
        # divided differences exp(-i lambda); stable at repeated eigenvalues
        dw=w[:,None]-w[None,:]
        dd=-1j*np.exp(-.5j*(w[:,None]+w[None,:]))*np.sinc(dw/(2*np.pi))
        gh=v@((v.conj().T@g@v)*dd.conj())@v.conj().T
        grad=np.real(np.einsum('ij,kij->k',gh.conj(),self.ps,optimize=True))/2
        return loss,grad
    def fit(self,likelihood):
        rng=np.random.default_rng(83177)
        starts=[self.x] if self.x is not None else [rng.normal(0,.025,len(self.ps)) for _ in range(self.restarts)]
        fits=[]
        for x in starts:
            f=minimize(self.objective,x,args=(likelihood,),jac=True,method='L-BFGS-B',options={'maxiter':self.maxiter,'gtol':1e-7,'ftol':1e-11,'maxcor':25,'maxls':30})
            fits.append(f)
        best=min(fits,key=lambda f:f.fun);self.x=best.x
        self.trace.append({'loss':float(best.fun),'iterations':int(best.nit),'converged':bool(best.success),'grad_inf':float(np.max(np.abs(best.jac)))})
        return self.matrix(self.x)

class GatePursuit:
    """Greedy insertion of rotations on any allowed local/pair Pauli, then joint MLE.
    Neither gate support nor order is supplied. BIC selects compact stopping point.
    """
    def __init__(self,n,max_gates=22,max_entanglers=8,beam=2):
        self.n=n;self.max_gates=max_gates;self.max_entanglers=max_entanglers;self.beam=beam
        self.labels=[];self.x=np.array([]);self.trace=[]
    def objective(self,x,labels,like):
        u,j,*_=circuit_and_jac(labels,x,self.n)
        loss,g=like(u)
        return loss,np.real(np.einsum('ij,kij->k',g.conj(),j,optimize=True))
    def refine(self,labels,x,like,maxiter=150):
        return minimize(self.objective,x,args=(labels,like),jac=True,method='L-BFGS-B',bounds=[(-np.pi,np.pi)]*len(labels),options={'maxiter':maxiter,'gtol':2e-7,'ftol':1e-10,'maxcor':15,'maxls':25})
    def fit(self,like):
        if self.labels:
            f=self.refine(self.labels,self.x,like);self.x=f.x
        ndata=like.total; penalty=np.log(ndata) # BIC deviance per extra parameter
        while len(self.labels)<self.max_gates:
            u,j,prefix,suffix=circuit_and_jac(self.labels,self.x,self.n)
            base,g=like(u); dic,mats=dictionary(self.n)
            ent=sum(sum(x!='I' for x in p)>1 for p in self.labels)
            candidates=[]
            for pos in range(len(self.labels)+1):
                for p,mat in zip(dic,mats):
                    if sum(x!='I' for x in p)>1 and ent>=self.max_entanglers:continue
                    if pos>0 and self.labels[pos-1]==p:continue
                    if pos<len(self.labels) and self.labels[pos]==p:continue
                    tangent=suffix[pos]@(-.5j*mat)@prefix[pos]
                    der=np.real(np.vdot(g,tangent))
                    candidates.append((abs(der),pos,p,der))
            # Avoid fitting many adjacent placements with same effective tangent.
            candidates.sort(reverse=True); trials=[];seen=set()
            for _,pos,p,der in candidates:
                if p in seen:continue
                seen.add(p)
                labels=self.labels[:pos]+[p]+self.labels[pos:]
                x=np.insert(self.x,pos,-np.sign(der)*.08)
                f=self.refine(labels,x,like)
                trials.append((f.fun,labels,f.x,f))
                if len(trials)>=self.beam:break
            if not trials:break
            loss,labels,x,f=min(trials,key=lambda x:x[0])
            gain=2*ndata*(base-loss)
            if gain<penalty:break
            self.labels=labels;self.x=x
        u,*_=circuit_and_jac(self.labels,self.x,self.n)
        self.trace.append({'loss':like(u)[0],'gates':len(self.labels),'entanglers':sum(sum(x!='I' for x in p)>1 for p in self.labels)})
        return u
    def circuit(self):return [{'pauli':p,'angle':float(t)} for p,t in zip(self.labels,self.x)]

class MatrixDistance:
    def __init__(self,target,total):self.target=target;self.total=total
    def __call__(self,u):
        s=np.vdot(self.target,u);d=u.shape[0]
        return float(1-abs(s)**2/d**2), -2*s*self.target/d**2

class DenseThenCompact:
    """Generic MLE -> gate-based compression -> observed-count refit.
    This is a strengthened *structural* control, not an unrestricted model.
    """
    def __init__(self,n):self.n=n;self.dense=DenseMLE(n);self.trace=[];self.compact=None
    def fit(self,like,dense_estimate=None):
        target=self.dense.fit(like) if dense_estimate is None else dense_estimate
        compact=GatePursuit(self.n)
        compact.fit(MatrixDistance(target,like.total))
        f=compact.refine(compact.labels,compact.x,like)
        compact.x=f.x;self.compact=compact
        u,*_=circuit_and_jac(compact.labels,compact.x,self.n)
        self.trace.append({'loss':like(u)[0],'gates':len(compact.labels),'entanglers':sum(sum(x!='I' for x in p)>1 for p in compact.labels)})
        return u
    def circuit(self):return self.compact.circuit()

class RidgeDenseMLE(DenseMLE):
    """Rotationally isotropic log-unitary shrinkage, lambda selected by count CV."""
    def __init__(self,n,maxiter=350):
        super().__init__(n,maxiter=maxiter,restarts=2);self.lam=0
    def objective(self,x,like):
        loss,grad=super().objective(x,like)
        return loss+self.lam*np.dot(x,x)/2,grad+self.lam*x
    def fit(self,like,init=None):
        rng=np.random.default_rng(12991)
        # Binomial thinning uses no additional oracle shots.
        train_counts=rng.binomial(like.counts.astype(int),.8)
        test_counts=like.counts-train_counts
        train=Likelihood(like.kets,like.bras,train_counts);valid=Likelihood(like.kets,like.bras,test_counts)
        x0=self.x if self.x is not None else init
        if x0 is None:x0=rng.normal(0,.025,len(self.ps))
        options=[]
        for lam in [0,.0003,.001,.003,.01,.03,.1]:
            self.lam=lam
            f=minimize(self.objective,x0,args=(train,),jac=True,method='L-BFGS-B',options={'maxiter':self.maxiter,'gtol':1e-7,'ftol':1e-11,'maxcor':25})
            options.append((valid(self.matrix(f.x))[0],lam,f.x))
        _,lam,x0=min(options,key=lambda t:t[0]);self.lam=lam
        f=minimize(self.objective,x0,args=(like,),jac=True,method='L-BFGS-B',options={'maxiter':self.maxiter,'gtol':1e-7,'ftol':1e-11,'maxcor':25})
        self.x=f.x
        self.trace.append({'loss':like(self.matrix(f.x))[0],'lambda':float(lam),'iterations':int(f.nit),'converged':bool(f.success),'grad_inf':float(np.max(np.abs(f.jac)))})
        return self.matrix(f.x)
