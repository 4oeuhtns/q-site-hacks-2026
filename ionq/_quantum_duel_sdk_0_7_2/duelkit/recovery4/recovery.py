"""Participant-only recovery. Imports no generator, hidden instance or private judge.
The block-aware method tries a dictionary of legal conjugation motifs in addition to
unstructured-to-compact projection and direct gate pursuit. No family IDs are supplied.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations
from functools import lru_cache
from scipy.optimize import minimize
from .quantum import *
from .legacy_learners import DenseMLE,GatePursuit,MatrixDistance

CAPS={'loose':(24,8),'medium':(18,6),'tight':(12,4)}
def deviance(like,u,k):return 2*like.total*like(u)[0]+k*np.log(like.total)
def entcount(gs):return sum(sum(a!='I' for a in g['pauli'])>1 for g in gs)

@lru_cache(None)
def motifs():
 """All shared-shell conjugations of two disjoint interactions; no instance input."""
 dic,mats=dictionary(4);inter=list(zip(dic[12:],mats[12:]));out=[];seen=set()
 for shell,S in inter:
  W=rotation(shell,np.pi/2)
  for (p,P),(q,Q) in combinations(inter,2):
   sp={j for j,a in enumerate(p) if a!='I'};sq={j for j,a in enumerate(q) if a!='I'}
   if sp & sq:continue
   if np.linalg.norm(S@P-P@S)<1 or np.linalg.norm(S@Q-Q@S)<1:continue
   Pe=W.conj().T@P@W;Qe=W.conj().T@Q@W
   key=(tuple(np.round(Pe.real,8).ravel()),tuple(np.round(Pe.imag,8).ravel()),tuple(np.round(Qe.real,8).ravel()),tuple(np.round(Qe.imag,8).ravel()))
   if key in seen:continue
   seen.add(key);out.append((shell,p,q,Pe,Qe))
 return tuple(out)

class FrameFit:
 """Shared entangling frame plus inferred sparse local conjugations.
 One local parameter costs TWO physical gates. Generator locations are searched.
 """
 def __init__(self,core,frame=(),x=None):
  self.core=core;self.frame=list(frame);self.x=np.array(x if x is not None else [0.,0.,np.pi/2],float)
 def physical(self,x=None,frame=None):
  if x is None:x=self.x
  if frame is None:frame=self.frame
  shell,p,q=self.core
  gs=[{'pauli':s,'angle':float(t)} for s,t in zip(frame,x[3:])]
  return gs+[{'pauli':shell,'angle':float(x[2])},{'pauli':p,'angle':float(x[0])},{'pauli':q,'angle':float(x[1])},{'pauli':shell,'angle':-float(x[2])}]+inverse(gs)
 def objective(self,x,frame,like):
  gs=self.physical(x,frame);ls=[g['pauli'] for g in gs];ts=[g['angle'] for g in gs]
  u,J,*_=circuit_and_jac(ls,ts,4);v,g=like(u);r=len(frame)
  D=[J[r+1],J[r+2],J[r]-J[r+3]]+[J[j]-J[len(gs)-1-j] for j in range(r)]
  return v,np.real(np.einsum('ij,kij->k',g.conj(),np.array(D),optimize=True))
 def refine(self,like,maxiter=220):
  f=minimize(self.objective,self.x,args=(self.frame,like),jac=True,method='L-BFGS-B',bounds=[(-np.pi,np.pi)]*len(self.x),options={'maxiter':maxiter,'gtol':2e-7,'ftol':1e-11,'maxls':30})
  self.x=f.x;return f.fun
 def grow(self,like,max_gates):
  self.refine(like)
  limit=min(4,(max_gates-4)//2)
  while len(self.frame)<limit:
   base=like(unitary(self.physical(),4))[0];trials=[];grads=[]
   for p in dictionary(4)[0][:12]:
    if p in self.frame:continue
    frame=[p]+self.frame;x=np.r_[self.x[:3],0.,self.x[3:]]
    grad=self.objective(x,frame,like)[1][3];grads.append((abs(grad),p,grad))
   for _,p,grad in sorted(grads,reverse=True)[:3]:
    trial=FrameFit(self.core,[p]+self.frame,np.r_[self.x[:3],-.2*np.sign(grad),self.x[3:]])
    v=trial.refine(like);trials.append((v,trial))
   if not trials:break
   v,best=min(trials,key=lambda t:t[0])
   if 2*like.total*(base-v)<2*np.log(like.total):break
   self.frame=best.frame;self.x=best.x
  return self

def frame_candidates(target,like,max_gates,top=5,previous=None):
 opts=[];tr=complex(np.trace(target)/16)
 phase=tr/abs(tr) if abs(tr)>1e-9 else 1.;target=target/phase
 for shell,p,q,Pe,Qe in motifs():
  a=2*np.arctan2(-float((np.trace(Pe@target)/16).imag),float(np.trace(target).real/16))
  b=2*np.arctan2(-float((np.trace(Qe@target)/16).imag),float(np.trace(target).real/16))
  f=FrameFit((shell,p,q),x=[a,b,np.pi/2]);u=unitary(f.physical(),4)
  opts.append((1-abs(np.vdot(target,u))**2/256,f))
 opts.sort(key=lambda t:t[0]);out=[]
 if previous is not None:
  prev=FrameFit(previous.core,previous.frame,previous.x.copy());prev.refine(like);out.append(prev)
 for _,f in opts[:top]:
  f.grow(MatrixDistance(target,like.total),max_gates)
  f.refine(like);out.append(f)
 return sorted(out,key=lambda f:deviance(like,unitary(f.physical(),4),len(f.physical())))

class Ensemble:
 def __init__(self,cap='medium',block=True,max_gates=None,max_entanglers=None):
  mg,me=CAPS[cap];mg=mg if max_gates is None else max_gates;me=me if max_entanglers is None else max_entanglers;self.mg=mg;self.me=me;self.block=block
  self.dense=DenseMLE(4,maxiter=700,restarts=4)
  self.direct=GatePursuit(4,max_gates=min(22,mg),max_entanglers=me,beam=2)
  self.proj=None;self.frame=None;self.history=[]
 def fit(self,like):
  target=self.dense.fit(like)
  options={}
  self.direct.fit(like);options['direct']=self.direct.circuit()
  pr=GatePursuit(4,max_gates=min(22,self.mg),max_entanglers=self.me,beam=2)
  pr.fit(MatrixDistance(target,like.total))
  if pr.labels:
   f=pr.refine(pr.labels,pr.x,like);pr.x=f.x
  options['projection']=pr.circuit();self.proj=pr
  bic={k:deviance(like,unitary(gs,4),len(gs)) for k,gs in options.items()}
  compact_name=min(bic,key=bic.get)
  if self.block:
   fs=frame_candidates(target,like,self.mg,previous=self.frame)
   self.frame=fs[0];options['block']=self.frame.physical();bic['block']=deviance(like,unitary(options['block'],4),len(options['block']))
  chosen=min(bic,key=bic.get)
  entry={'compact':inverse(options[compact_name]),'block':inverse(options[chosen]),'selected_compact':compact_name,'selected_block':chosen,'bic':bic,
         'candidates':{k:inverse(v) for k,v in options.items()},'dense_trace':self.dense.trace[-1]}
  self.history.append(entry);return entry
