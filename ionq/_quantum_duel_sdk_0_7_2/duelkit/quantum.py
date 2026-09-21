"""Dimension-aware 2/4-qubit kernel and version-stable product-Pauli experiment IDs.

Server paths always pass Rules.qubits explicitly. Default n=2 retains legacy
public SDK compatibility, not a license to infer a circuit's register size.
4q measurement catalogs are lazy: never allocate 104,976 dense basis matrices.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
from functools import lru_cache
from itertools import product
import math, hashlib, json
import numpy as np
from scipy.linalg import expm
from . import legacy_quantum as legacy

I2=legacy.I2; I4=legacy.I4; X=legacy.X; Y=legacy.Y; Z=legacy.Z; H=legacy.H
PAULI=legacy.PAULI; STATE=legacy.STATE
Gate=legacy.Gate; G=legacy.G; Circuit=legacy.Circuit
SINGLE=legacy.SINGLE; DOUBLE=legacy.DOUBLE; ROTATIONS=legacy.ROTATIONS
canonical_json=legacy.canonical_json; digest=legacy.digest; derive_seed=legacy.derive_seed
circuit_data=legacy.circuit_data
LABELS=legacy.LABELS; PMATS=legacy.PMATS; LABEL_INDEX=legacy.LABEL_INDEX
# Legacy table aliases remain small and stable for historical public solvers.
EXPERIMENTS=legacy.EXPERIMENTS; EXPERIMENT_INDEX=legacy.EXPERIMENT_INDEX
KETS=legacy.KETS; MEAS_BASIS=legacy.MEAS_BASIS
TOMOGRAPHY_PANEL=legacy.TOMOGRAPHY_PANEL; BITSTRINGS=legacy.BITSTRINGS

def _n(n):
    if type(n) is not int or n not in (2,4,8):
        raise ValueError("Supported registers are 2, 4 and 8 qubits")
    return n

def from_data(data):
    if not isinstance(data,list) or len(data)>108:
        raise ValueError("Circuit must be a list of at most 108 gate records")
    out=[]
    for v in data:
        if not isinstance(v,dict) or set(v)!={"name","targets","angle"}:
            raise ValueError("Only name, targets and angle are allowed")
        if not isinstance(v["targets"],list): raise ValueError("targets must be an array")
        out.append(Gate(v["name"],tuple(v["targets"]),v["angle"]))
    return tuple(out)

def validate(circuit, *, n=2, max_gates=24, max_entanglers=3,
             alphabet=None, max_angle=8*math.pi):
    _n(n)
    allowed=SINGLE|DOUBLE if alphabet is None else set(alphabet)
    if not isinstance(circuit,tuple): raise ValueError("Use an immutable tuple of Gate records")
    if len(circuit)>max_gates: raise ValueError(f"Too many gates: {len(circuit)}; limit {max_gates}")
    for g in circuit:
        if not isinstance(g,Gate) or g.name not in allowed or g.name not in SINGLE|DOUBLE:
            raise ValueError("Gate is not in this round's static alphabet")
        arity=1 if g.name in SINGLE else 2
        if (len(g.targets)!=arity or len(set(g.targets))!=arity or
            any(type(i) is not int or not 0<=i<n for i in g.targets)):
            raise ValueError(f"Gate needs {arity} distinct targets in [0,{n-1}]")
        if g.name in ROTATIONS:
            if isinstance(g.angle,(bool,str)) or not isinstance(g.angle,(int,float,np.floating)):
                raise ValueError("Rotation angle must be a finite JSON number in radians")
            if not np.isfinite(g.angle) or abs(g.angle)>max_angle:
                raise ValueError("Rotation angle outside the round's canonical range")
        elif g.angle is not None: raise ValueError("This gate has no angle parameter")
    ent=sum(g.name in DOUBLE for g in circuit)
    if ent>max_entanglers: raise ValueError(f"Too many abstract entanglers: {ent}; limit {max_entanglers}")
    return {"gates":len(circuit),"abstract_entanglers":ent,"single_qubit_gates":len(circuit)-ent}

def kron_all(ops):
    out=np.array([[1]],complex)
    for op in ops: out=np.kron(out,op)
    return out

@lru_cache(maxsize=1024)
def generator(label):
    if any(x not in PAULI for x in label): raise ValueError("Invalid Pauli string")
    m=kron_all([PAULI[c] for c in label]); m.flags.writeable=False
    return m

def gate_matrix(g,n=2):
    _n(n)
    if n==2: return legacy.gate_matrix(g)
    d=1<<n
    if g.name in ROTATIONS:
        p=generator(''.join(g.name[-1].upper() if i in g.targets else 'I' for i in range(n)))
        return np.cos(g.angle/2)*np.eye(d)-1j*np.sin(g.angle/2)*p
    if g.name in ('x','y','z','h'):
        m=H if g.name=='h' else PAULI[g.name.upper()]
        return kron_all([m if i==g.targets[0] else I2 for i in range(n)])
    if g.name=='cx':
        out=np.zeros((d,d),complex)
        for col in range(d):
            row=col ^ (1<<(n-1-g.targets[1])) if col & (1<<(n-1-g.targets[0])) else col
            out[row,col]=1
        return out
    raise ValueError("Unsupported gate")

def unitary(circuit,n=2):
    _n(n)
    if n==2: return legacy.unitary(circuit)
    u=np.eye(1<<n,dtype=complex)
    if n==8:
        for g in circuit:u=apply_gate(g,u,n)
        return u
    for g in circuit: u=gate_matrix(g,n)@u
    return u

def inverse(circuit): return legacy.inverse(circuit)

def independent_unitary(circuit,n=2):
    _n(n)
    if n==2: return legacy.independent_unitary(circuit)
    if n==8:
        # Independent embedding: exponentiate ONLY 2x2/4x4 matrices and apply by tensor axes.
        u=np.eye(1<<n,dtype=complex)
        for g in circuit:
            k=len(g.targets)
            a=PAULI[g.name[-1].upper()]
            local=expm(-0.5j*float(g.angle)*(a if k==1 else np.kron(a,a)))
            order=list(g.targets)+[q for q in range(n) if q not in g.targets]+[n]
            tensor=u.reshape([2]*n+[1<<n]).transpose(order)
            tensor=(local@tensor.reshape(1<<k,-1)).reshape(tensor.shape)
            u=tensor.transpose(np.argsort(order)).reshape(1<<n,1<<n)
        return u
    # Separate direct tensor/exponential path; no gate_matrix/generator reuse.
    d=1<<n; out=np.eye(d,dtype=complex)
    for g in circuit:
        factors=[np.eye(2,dtype=complex) for _ in range(n)]
        if g.name in ROTATIONS:
            for t in g.targets: factors[t]=PAULI[g.name[-1].upper()]
            p=kron_all(factors); m=expm(-0.5j*float(g.angle)*p)
        elif g.name=='cx':
            a=list(factors); b=list(factors)
            a[g.targets[0]]=(I2+Z)/2; b[g.targets[0]]=(I2-Z)/2; b[g.targets[1]]=X
            m=kron_all(a)+kron_all(b)
        else:
            factors[g.targets[0]]=H if g.name=='h' else PAULI[g.name.upper()]
            m=kron_all(factors)
        out=m@out
    return out

def _shape(u,target=None):
    u=np.asarray(u,complex)
    if u.ndim!=2 or u.shape[0]!=u.shape[1] or u.shape[0] not in (4,16,256) or not np.isfinite(u).all():
        raise ValueError("Expected a finite 4x4, 16x16 or 256x256 matrix")
    target=np.eye(u.shape[0],dtype=complex) if target is None else np.asarray(target,complex)
    if target.shape!=u.shape or not np.isfinite(target).all(): raise ValueError("Target dimension mismatch")
    return u,target

def infidelity(u,target=None):
    u,target=_shape(u,target)
    overlap=np.vdot(target,u) if len(u)==256 else np.trace(target.conj().T@u)
    return float(np.clip(1-abs(overlap)**2/u.shape[0]**2,0,1))

def choi_infidelity(u,target=None):
    u,target=_shape(u,target);d=u.shape[0]
    return float(np.clip(1-abs(np.vdot(target.reshape(-1)/np.sqrt(d),u.reshape(-1)/np.sqrt(d)))**2,0,1))

def global_distance(u,v):
    u,v=_shape(u,v)
    overlap=np.vdot(u,v) if len(u)==256 else np.trace(u.conj().T@v)
    return float(np.sqrt(max(0,2-2*abs(overlap)/u.shape[0])))

def ptm(u):
    u,_=_shape(u);n=int(round(np.log2(u.shape[0])))
    if n>4:raise ValueError('Full PTM disabled for eight qubits; use measured residuals')
    mats=PMATS if n==2 else np.array([generator(''.join(p)) for p in product('IXYZ',repeat=n)])
    ev=np.einsum('ab,jbc,cd->jad',u,mats,u.conj().T)
    return np.einsum('iab,jba->ij',mats,ev).real/u.shape[0]

@dataclass(frozen=True)
class Experiment:
    prep: tuple[str,...]
    basis: str
    def data(self): return {"prep":list(self.prep),"basis":self.basis}

PREPARATIONS=tuple(STATE)  # Z+, Z-, X+, X-, Y+, Y-; legacy index order retained.
PREP_ALIASES={"0":"Z+","1":"Z-","+":"X+","-":"X-","+i":"Y+","-i":"Y-"}

def bitstrings(n): return tuple(format(i,f'0{_n(n)}b') for i in range(1<<n))

def _digits(k,base,n):
    out=[0]*n
    for j in range(n-1,-1,-1): out[j]=k%base; k//=base
    return out

def experiment_at(index,n=2):
    _n(n)
    if type(index) is not int or not 0<=index<18**n: raise ValueError("Experiment index outside the round's register")
    a,b=divmod(index,3**n)
    return Experiment(tuple(PREPARATIONS[x] for x in _digits(a,6,n)),''.join('XYZ'[x] for x in _digits(b,3,n)))

def experiment_index(prep,basis,n=2):
    _n(n)
    if len(prep)!=n or len(basis)!=n: raise ValueError("Preparation and readout must match register length")
    a=b=0
    for x in prep:
        x=PREP_ALIASES.get(x,x)
        if x not in PREPARATIONS: raise ValueError("Unknown Pauli eigenstate")
        a=6*a+PREPARATIONS.index(x)
    for x in basis:
        if x not in 'XYZ': raise ValueError("Readout must use X, Y or Z")
        b=3*b+'XYZ'.index(x)
    return a*3**n+b

class ExperimentCatalog(Sequence):
    def __init__(self,n): self.n=_n(n)
    def __len__(self): return 18**self.n
    def __getitem__(self,i):
        if isinstance(i,slice):
            rr=range(*i.indices(len(self)))
            if len(rr)>2048:raise ValueError('Use bounded catalogue pages')
            return [self[j] for j in rr]
        return experiment_at(i,self.n)

def experiment_catalog(n): return ExperimentCatalog(n)

@lru_cache(maxsize=2048)
def experiment_arrays(index,n):
    if n==8:raise ValueError('Eight-qubit readout bases must use the streamed path')
    e=experiment_at(index,n)
    ket=kron_all([STATE[x].reshape(2,1) for x in e.prep]).reshape(-1)
    basis=kron_all([np.column_stack([STATE[x+'+'],STATE[x+'-']]) for x in e.basis])
    ket.flags.writeable=False;basis.flags.writeable=False
    return ket,basis

def probabilities(u,indices=None):
    u,_=_shape(u);n=int(round(np.log2(u.shape[0])))
    if n==2: return legacy.probabilities(u,indices)
    if n==8:
        if indices is None:raise ValueError('Explicit bounded indices required')
        ids=list(indices)
        if len(ids)>2048:raise ValueError('Probability batch exceeds 2048 settings')
        return np.asarray([stream_probability(u,int(i),n) for i in ids]).reshape(len(ids),1<<n)
    if indices is None: raise ValueError("Explicit indices required: four-qubit catalog is lazy")
    ids=list(indices)
    if len(ids)>2048: raise ValueError("Probability batch exceeds 2048 settings")
    if not ids:return np.zeros((0,1<<n))
    arrays=[experiment_arrays(int(i),n) for i in ids]
    ket=np.stack([a for a,b in arrays]);basis=np.stack([b for a,b in arrays])
    state=np.einsum('ab,sb->sa',u,ket)
    amplitude=np.einsum('sba,sb->sa',basis.conj(),state)
    p=np.clip(np.abs(amplitude)**2,0,1)
    return p/p.sum(axis=1,keepdims=True)

def independent_probabilities(u,index,n):
    e=experiment_at(index,n)
    rho=kron_all([np.outer(STATE[x],STATE[x].conj()) for x in e.prep])
    rho=u@rho@u.conj().T
    values=[]
    for bits in bitstrings(n):
        projector=kron_all([(I2+(1 if b=='0' else -1)*PAULI[a])/2 for a,b in zip(e.basis,bits)])
        values.append(float(np.trace(projector@rho).real))
    p=np.maximum(0,values);return p/p.sum()

def outcome_signs(mask,n=2):
    if type(mask) is not int or not 0<=mask<(1<<n):raise ValueError("Invalid observable mask")
    return np.array([(-1)**((i&mask).bit_count()) for i in range(1<<n)])

def expectation_from_counts(counts,mask=None):
    if not counts:raise ValueError("No counts")
    n=len(next(iter(counts)));bits=bitstrings(n);mask=(1<<n)-1 if mask is None else mask
    c=np.array([counts.get(s,0) for s in bits],float)
    if not np.isfinite(c).all() or np.any(c<0) or c.sum()<=0:raise ValueError("Invalid counts")
    return float(c@outcome_signs(mask,n)/c.sum())

def wilson_expectation(counts,mask=None,z=1.959963984540054):
    if not counts:raise ValueError("No counts")
    n=len(next(iter(counts)));bits=bitstrings(n);mask=(1<<n)-1 if mask is None else mask
    c=np.array([counts.get(s,0) for s in bits],float);N=c.sum()
    if N<=0 or np.any(c<0):raise ValueError("Positive nonnegative counts required")
    p=c[outcome_signs(mask,n)>0].sum()/N;den=1+z*z/N
    center=(p+z*z/(2*N))/den;half=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/den
    return float(2*max(0,center-half)-1),float(2*min(1,center+half)-1)

# Original teaching rank utility retains its explicitly two-qubit meaning.
experiment_rank=legacy.experiment_rank if hasattr(legacy,'experiment_rank') else None
measurement_rank=legacy.measurement_rank if hasattr(legacy,'measurement_rank') else None

def qasm_export(circuit,n=2,**limits):
    validate(circuit,n=n,**limits)
    lines=['OPENQASM 3.0;','include "stdgates.inc";',f'qubit[{n}] q;']
    def emit(name,qs,a=None):
        angle='' if a is None else '('+format(float(a),'.17g')+')'
        lines.append(name+angle+' '+', '.join(f'q[{q}]' for q in qs)+';')
    for g in circuit:
        if g.name in ('rxx','ryy','rzz'):
            a,b=g.targets
            if g.name=='rxx':emit('h',(a,));emit('h',(b,))
            if g.name=='ryy':emit('rx',(a,),np.pi/2);emit('rx',(b,),np.pi/2)
            emit('cx',(a,b));emit('rz',(b,),g.angle);emit('cx',(a,b))
            if g.name=='rxx':emit('h',(a,));emit('h',(b,))
            if g.name=='ryy':emit('rx',(a,),-np.pi/2);emit('rx',(b,),-np.pi/2)
        else:emit(g.name,g.targets,g.angle)
    return '\n'.join(lines)+'\n'

# Compatibility exports for old public notebooks only.
for _name in ("tomography_rank", "table_rows", "CircuitBuilder"):
    if hasattr(legacy,_name) and _name not in globals(): globals()[_name]=getattr(legacy,_name)


@lru_cache(maxsize=128)
def _pauli_rows(name,targets,n):
    xmask=sum(1<<(n-1-q) for q in targets) if name[-1] in 'xy' else 0
    ids=np.arange(1<<n);rows=ids^xmask;phase=np.ones(1<<n,dtype=complex)
    for q in targets:
        bit=(rows>>(n-1-q))&1
        if name[-1]=='y':phase*=1j*(1-2*bit)
        elif name[-1]=='z':phase*=1-2*bit
    return rows,phase

def apply_gate(g,state,n):
    if g.name not in ROTATIONS:raise ValueError('Eight-qubit profile only supports rotations')
    rows,phase=_pauli_rows(g.name,tuple(g.targets),n)
    scale=phase if state.ndim==1 else phase[:,None]
    return np.cos(g.angle/2)*state-1j*np.sin(g.angle/2)*scale*state[rows]

def apply_circuit(circuit,state,n):
    for g in circuit:state=apply_gate(g,state,n)
    return state

@lru_cache(maxsize=512)
def prepared_state(index,n):
    e=experiment_at(index,n)
    v=kron_all([STATE[x].reshape(2,1) for x in e.prep]).reshape(-1)
    v.flags.writeable=False
    return v

def stream_probability(u,index,n=8,analysis=()):
    e=experiment_at(index,n)
    state=u@prepared_state(index,n)
    state=apply_circuit(analysis,state,n)
    t=state.reshape([2]*n)
    for q,b in enumerate(e.basis):
        m=np.column_stack([STATE[b+'+'],STATE[b+'-']]).conj().T
        t=np.moveaxis(np.tensordot(m,t,axes=(1,q)),0,q)
    p=np.maximum(0,np.abs(t.reshape(-1))**2)
    return p/p.sum()
