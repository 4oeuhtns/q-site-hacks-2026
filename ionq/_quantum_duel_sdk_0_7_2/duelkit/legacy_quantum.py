"""Two-qubit teaching kernel. Static circuits; q0 is the LEFT tensor factor.

No credentials, network requests or participant-code execution. This is not a
production circuit sandbox. Angles in the circuit IR are always radians.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
from functools import lru_cache
import hashlib, json, math
import numpy as np
from scipy.linalg import expm

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.diag([1,-1]).astype(complex)
H = (X+Z)/np.sqrt(2)
PAULI = dict(I=I2, X=X, Y=Y, Z=Z)
LABELS = tuple(''.join(p) for p in product('IXYZ', repeat=2))
PMATS = np.array([np.kron(PAULI[p[0]], PAULI[p[1]]) for p in LABELS])
LABEL_INDEX = {p:i for i,p in enumerate(LABELS)}
BITSTRINGS = ('00','01','10','11')


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def derive_seed(*parts):
    return int.from_bytes(hashlib.sha256(canonical_json(parts).encode()).digest()[:8], 'big')


@dataclass(frozen=True)
class Gate:
    name: str
    targets: tuple[int, ...]
    angle: float | None = None

    def data(self):
        return {'name':self.name, 'targets':list(self.targets), 'angle':self.angle}


Circuit = tuple[Gate, ...]
SINGLE = {'rx','ry','rz','x','y','z','h'}
DOUBLE = {'rxx','ryy','rzz','cx'}
ROTATIONS = {'rx','ry','rz','rxx','ryy','rzz'}


def G(name, target=0, angle=None):
    return Gate(name, (target,) if isinstance(target, int) else tuple(target), angle)


def circuit_data(circuit):
    return [g.data() for g in circuit]


def from_data(data):
    if not isinstance(data, list) or len(data)>64:
        raise ValueError('Circuit must be a list of at most 64 gate objects.')
    out=[]
    for v in data:
        if not isinstance(v, dict) or set(v) != {'name','targets','angle'}:
            raise ValueError('Only name, targets and angle fields are allowed.')
        out.append(Gate(v['name'], tuple(v['targets']), v['angle']))
    return tuple(out)


def validate(circuit, *, max_gates=24, max_entanglers=3):
    if not isinstance(circuit, tuple):
        raise ValueError('Use an immutable tuple of Gate records.')
    if len(circuit)>max_gates:
        raise ValueError(f'Too many gates: {len(circuit)}; limit {max_gates}.')
    for g in circuit:
        if not isinstance(g, Gate) or g.name not in SINGLE | DOUBLE:
            raise ValueError('Only the published static gate grammar is allowed.')
        arity=1 if g.name in SINGLE else 2
        if (len(g.targets)!=arity or len(set(g.targets))!=arity
            or any(type(q) is not int or q not in (0,1) for q in g.targets)):
            raise ValueError('Gate targets must be distinct qubits 0 and/or 1.')
        if g.name in ROTATIONS:
            if isinstance(g.angle, (bool, str)) or not isinstance(g.angle, (int,float,np.floating)):
                raise ValueError('A rotation needs one finite numeric angle in radians.')
            if not np.isfinite(g.angle) or abs(g.angle)>8*np.pi:
                raise ValueError('Angle is nonfinite or outside the draft numerical range.')
        elif g.angle is not None:
            raise ValueError('This gate has no angle parameter.')
    ent=sum(g.name in DOUBLE for g in circuit)
    if ent>max_entanglers:
        raise ValueError(f'Too many abstract entanglers: {ent}; limit {max_entanglers}.')
    return {'gates':len(circuit),'abstract_entanglers':ent,
            'single_qubit_gates':len(circuit)-ent}


def gate_matrix(g):
    if g.name in ('rx','ry','rz'):
        p=PAULI[g.name[-1].upper()]
        m=np.cos(g.angle/2)*I2-1j*np.sin(g.angle/2)*p
        return np.kron(m,I2) if g.targets[0]==0 else np.kron(I2,m)
    if g.name in ('rxx','ryy','rzz'):
        p=np.kron(PAULI[g.name[-1].upper()], PAULI[g.name[-1].upper()])
        return np.cos(g.angle/2)*I4-1j*np.sin(g.angle/2)*p
    if g.name in ('x','y','z','h'):
        m=H if g.name=='h' else PAULI[g.name.upper()]
        return np.kron(m,I2) if g.targets[0]==0 else np.kron(I2,m)
    if g.name=='cx':
        a=np.zeros((4,4),complex)
        for j in range(4):
            b=[j//2,j%2]
            b[g.targets[1]] ^= b[g.targets[0]]
            a[2*b[0]+b[1],j]=1
        return a
    raise ValueError('Unsupported gate')


def unitary(circuit):
    u=I4.copy()
    for g in circuit: # left-to-right TIME order; matrices multiply on the left
        u=gate_matrix(g)@u
    return u


def inverse(circuit):
    return tuple(Gate(g.name,g.targets,-float(g.angle) if g.name in ROTATIONS else None)
                 for g in reversed(circuit))


def independent_unitary(circuit):
    """Separate expm/controlled-projector path used by conformance tests."""
    u=I4.copy()
    for g in circuit:
        if g.name in ROTATIONS:
            if len(g.targets)==1:
                p=np.kron(PAULI[g.name[-1].upper()], I2) if g.targets[0]==0 else np.kron(I2,PAULI[g.name[-1].upper()])
            else:
                p=np.kron(PAULI[g.name[-1].upper()],PAULI[g.name[-1].upper()])
            m=expm(-0.5j*float(g.angle)*p)
        elif g.name=='cx':
            p0=(I2+Z)/2; p1=(I2-Z)/2
            m=np.kron(p0,I2)+np.kron(p1,X) if g.targets==(0,1) else np.kron(I2,p0)+np.kron(X,p1)
        else:
            a=H if g.name=='h' else PAULI[g.name.upper()]
            m=np.kron(a,I2) if g.targets[0]==0 else np.kron(I2,a)
        u=m@u
    return u


def infidelity(u, target=I4):
    return float(np.clip(1-abs(np.trace(target.conj().T@u))**2/16,0,1))


def choi_infidelity(u, target=I4):
    a=np.asarray(u).reshape(-1)/2
    b=np.asarray(target).reshape(-1)/2
    return float(np.clip(1-abs(np.vdot(b,a))**2,0,1))


def ptm(u):
    evolved=np.einsum('ab,jbc,cd->jad',u,PMATS,u.conj().T)
    return np.einsum('iab,jba->ij',PMATS,evolved).real/4


def global_distance(u,v):
    return float(np.sqrt(max(0,2-2*abs(np.trace(u.conj().T@v))/4)))


STATE={
    'Z+':np.array([1,0],complex), 'Z-':np.array([0,1],complex),
    'X+':np.array([1,1],complex)/np.sqrt(2), 'X-':np.array([1,-1],complex)/np.sqrt(2),
    'Y+':np.array([1,1j],complex)/np.sqrt(2), 'Y-':np.array([1,-1j],complex)/np.sqrt(2),
}


@dataclass(frozen=True)
class Experiment:
    prep: tuple[str,str]
    basis: str
    def data(self): return {'prep':list(self.prep),'basis':self.basis}


EXPERIMENTS=tuple(Experiment(tuple(p), ''.join(b))
                  for p in product(STATE, repeat=2) for b in product('XYZ',repeat=2))
EXPERIMENT_INDEX={e:i for i,e in enumerate(EXPERIMENTS)}
KETS=np.array([np.kron(STATE[e.prep[0]],STATE[e.prep[1]]) for e in EXPERIMENTS])
# Columns are the basis eigenvectors in bit order 00,01,10,11.
MEAS_BASIS=np.array([np.column_stack([np.kron(STATE[e.basis[0]+('+' if s[0]=='0' else '-')],
                                                        STATE[e.basis[1]+('+' if s[1]=='0' else '-')])
                                    for s in BITSTRINGS]) for e in EXPERIMENTS])
TOMOGRAPHY_PANEL=np.array([i for i,e in enumerate(EXPERIMENTS)
                         if all(p in ('Z+','Z-','X+','Y+') for p in e.prep)],dtype=int)


def probabilities(u, indices=None):
    ids=np.arange(len(EXPERIMENTS)) if indices is None else np.asarray(indices,dtype=int)
    states=np.einsum('ab,sb->sa',u,KETS[ids])
    amplitudes=np.einsum('sba,sb->sa',MEAS_BASIS[ids].conj(),states)
    p=np.abs(amplitudes)**2
    p=np.clip(p,0,1)
    return p/p.sum(axis=1,keepdims=True)


def outcome_signs(mask):
    if mask not in (0,1,2,3): raise ValueError('mask must be 0..3')
    return np.array([(-1)**sum(int(bits[q]) for q in (0,1) if mask & (2>>q)) for bits in BITSTRINGS])


def expectation_from_counts(counts, mask=3):
    c=np.array([counts.get(s,0) for s in BITSTRINGS],float)
    if c.sum()<=0: raise ValueError('Positive count total required.')
    return float(c@outcome_signs(mask)/c.sum())


def wilson_expectation(counts, mask=3, z=1.959963984540054):
    """Pointwise interval for a selected binary observable; not a simultaneous certificate."""
    c=np.array([counts.get(s,0) for s in BITSTRINGS],float)
    n=c.sum()
    if n<=0: raise ValueError('Positive count total required.')
    p=c[outcome_signs(mask)>0].sum()/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (float(2*max(0,center-half)-1),float(2*min(1,center+half)-1))


def measurement_design_rank():
    """Rank on the 16x16 real Pauli representation using raw probabilities."""
    rows=[]
    for i in TOMOGRAPHY_PANEL:
        rho=np.outer(KETS[i],KETS[i].conj())
        rin=np.einsum('iab,ba->i',PMATS,rho).real
        for b in range(4):
            v=MEAS_BASIS[i,:,b]
            eff=np.outer(v,v.conj())
            mout=np.einsum('iab,ba->i',PMATS,eff).real
            rows.append(np.outer(mout,rin).reshape(-1)/4)
    return int(np.linalg.matrix_rank(np.array(rows),tol=1e-9))


def qasm_export(circuit):
    """Static QASM transport only. Backend acceptance/native lowering are NOT certified."""
    validate(circuit)
    lines=['OPENQASM 3.0;','include "stdgates.inc";','qubit[2] q;']
    # Expand Pauli interactions into the narrow rx/ry/rz/h/cx transport basis.
    def emit(name,qs,a=None):
        angle='' if a is None else '('+format(float(a),'.17g')+')'
        lines.append(name+angle+' '+', '.join(f'q[{q}]' for q in qs)+';')
    for g in circuit:
        if g.name in ('rxx','ryy','rzz'):
            if g.name=='rxx':
                emit('h',(0,)); emit('h',(1,))
            if g.name=='ryy':
                emit('rx',(0,),np.pi/2); emit('rx',(1,),np.pi/2)
            emit('cx',(0,1)); emit('rz',(1,),g.angle); emit('cx',(0,1))
            if g.name=='rxx':
                emit('h',(0,)); emit('h',(1,))
            if g.name=='ryy':
                emit('rx',(0,),-np.pi/2); emit('rx',(1,),-np.pi/2)
        else: emit(g.name,g.targets,g.angle)
    return '\n'.join(lines)+'\n'
