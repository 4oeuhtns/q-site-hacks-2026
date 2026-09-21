"""Public 1--6 qubit quantum primitives. No instance access.
q0 is the left/MSB tensor factor; gate arrays are chronological; angles are radians.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import product, combinations
import numpy as np
from scipy.linalg import expm

I=np.eye(2,dtype=complex)
P1={'I':I,'X':np.array([[0,1],[1,0]],complex),'Y':np.array([[0,-1j],[1j,0]],complex),'Z':np.diag([1,-1]).astype(complex)}
KETS={'0':np.array([1,0],complex),'1':np.array([0,1],complex),'+':np.array([1,1],complex)/np.sqrt(2),'-':np.array([1,-1],complex)/np.sqrt(2),'+i':np.array([1,1j],complex)/np.sqrt(2),'-i':np.array([1,-1j],complex)/np.sqrt(2)}
BASES={'Z':I,'X':np.column_stack([KETS['+'],KETS['-']]),'Y':np.column_stack([KETS['+i'],KETS['-i']])}
def kron(xs):
    out=np.array([1.],complex)
    for x in xs:out=np.kron(out,x)
    return out
@lru_cache(None)
def pauli(label):return kron([P1[x] for x in label])
@lru_cache(None)
def all_paulis(n):
    labels=[''.join(x) for x in product('IXYZ',repeat=n)][1:]
    return labels,np.stack([pauli(x) for x in labels])
@lru_cache(None)
def dictionary(n):
    labels=[]
    for q in range(n):
        for a in 'XYZ':labels.append('I'*q+a+'I'*(n-q-1))
    for a,b in combinations(range(n),2):
        for p in 'XYZ':labels.append(''.join(p if q in (a,b) else 'I' for q in range(n)))
    return tuple(labels),np.stack([pauli(x) for x in labels])
def rotation(label,t):return np.cos(t/2)*np.eye(2**len(label))-1j*np.sin(t/2)*pauli(label)
def unitary(gates,n):
    u=np.eye(2**n,dtype=complex)
    for g in gates:u=rotation(g['pauli'],g['angle'])@u
    return u
def inverse(gates):return [{'pauli':g['pauli'],'angle':-float(g['angle'])} for g in gates[::-1]]
def error(a,p):return float(np.clip(1-abs(np.trace(p@a))**2/a.shape[0]**2,0,1))
def score(e):return float(100*np.clip(np.log10(.1/max(float(e),1e-15))/2,0,1))
def independent_unitary(gates,n):
    u=np.eye(2**n,dtype=complex)
    for g in gates:u=expm(-.5j*float(g['angle'])*pauli(g['pauli']))@u
    return u

def make_panel(n,settings,seed):
    # Product eigenstates and Pauli bases; fixed public randomized design.
    rng=np.random.default_rng(seed); seen=set();out=[]
    while len(out)<settings:
        prep=tuple(rng.choice(['0','1','+','-','+i','-i'],size=n))
        meas=tuple(rng.choice(list('XYZ'),size=n))
        key=(prep,meas)
        if key not in seen:seen.add(key);out.append({'prep':list(prep),'basis':list(meas)})
    return out

def design(panel):
    kets=np.stack([kron([KETS[a] for a in e['prep']]) for e in panel])
    bras=np.stack([kron([BASES[a] for a in e['basis']]).conj().T for e in panel])
    return kets,bras

def probs(u,kets,bras):
    evolved=kets@u.T
    amp=np.einsum('sij,sj->si',bras,evolved,optimize=True)
    return np.abs(amp)**2

class Likelihood:
    def __init__(self,kets,bras,counts):
        self.kets=kets;self.bras=bras;self.counts=np.asarray(counts,float);self.total=float(self.counts.sum())
    def __call__(self,u):
        evolved=self.kets@u.T
        amp=np.einsum('sij,sj->si',self.bras,evolved,optimize=True)
        p=np.abs(amp)**2 + 1e-12
        loss=-np.sum(self.counts*np.log(p))/self.total
        w=(self.counts/p)*amp
        back=np.einsum('sji,sj->si',self.bras.conj(),w,optimize=True)
        grad=-2*back.T@self.kets.conj()/self.total
        return float(loss),grad

def circuit_and_jac(labels,x,n):
    d=2**n;eye=np.eye(d,dtype=complex)
    gates=[rotation(p,t) for p,t in zip(labels,x)]
    prefix=[eye]
    for g in gates:prefix.append(g@prefix[-1])
    suffix=[None]*(len(gates)+1);suffix[-1]=eye
    for j in range(len(gates)-1,-1,-1):suffix[j]=suffix[j+1]@gates[j]
    deriv=np.stack([suffix[j+1]@(-.5j*pauli(p))@prefix[j+1] for j,p in enumerate(labels)]) if labels else np.zeros((0,d,d),complex)
    return prefix[-1],deriv,prefix,suffix
