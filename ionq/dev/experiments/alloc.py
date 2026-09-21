"""Answer the literal question: how much is optimal SHOT ALLOCATION worth?

Compare, at fixed N, the CRB floor eps = 0.5*Tr(M F(w)^-1) for
  uniform w over 180 settings        (what the stock adapter does)
  A-optimal w                        (an oracle that knows the attack -- upper bound)
  uniform over a 60-setting subset    (fewer settings, more shots each)
Plus: how much of the 180-setting budget is actually worth using.
"""
import sys, json
sys.path.insert(0,"/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/_quantum_duel_sdk_0_7_2")
import numpy as np
from duelkit.recovery4.quantum import make_panel, design, circuit_and_jac, unitary as u4
from qduel_sdk.contracts import Template
from qduel_sdk.rules import Rules
from qduel_sdk.templates import instantiate

R=Rules(); PANEL=make_panel(4,180,60231); KETS,BRAS=design(PANEL)

def labels_of(circ,n=4):
    return [''.join(g.name[-1].upper() if q in g.targets else 'I' for q in range(n)) for g in circ]

def per_setting_fisher(labels,x):
    """F_s for each setting s, shape (S,K,K)."""
    U,J,*_=circuit_and_jac(labels,x,4)
    amp=np.einsum('sij,sj->si',BRAS,KETS@U.T,optimize=True)
    damp=np.einsum('sij,kja,sa->ksi',BRAS,J,KETS,optimize=True)
    p=np.maximum(np.abs(amp)**2,1e-12)
    dp=2*np.real(np.conj(amp)[None]*damp)
    return np.einsum('ksi,lsi,si->skl',dp,dp,1/p,optimize=True)

def hess(labels,x):
    d=16; k=len(x); h=1e-4; M=np.zeros((k,k))
    def eps(dx):
        Uh=u4([{'pauli':p,'angle':t} for p,t in zip(labels,x+dx)],4)
        Ut=u4([{'pauli':p,'angle':t} for p,t in zip(labels,x)],4)
        return float(np.clip(1-abs(np.trace(Uh.conj().T@Ut))**2/d**2,0,1))
    for i in range(k):
        for j in range(k):
            e=np.zeros(k);f=np.zeros(k);e[i]=h;f[j]=h
            M[i,j]=(eps(e+f)-eps(e-f)-eps(-e+f)+eps(-e-f))/(4*h*h)
    return (M+M.T)/2

def obj(Fs,w,M,N):
    F=np.einsum('s,skl->kl',w,Fs)*N
    return 0.5*np.trace(M@np.linalg.inv(F+1e-12*np.eye(len(M))))

def a_optimal(Fs,M,N,iters=400):
    """Multiplicative weight updates for the L-optimal (weighted-A) design."""
    S=len(Fs); w=np.ones(S)/S
    for _ in range(iters):
        F=np.einsum('s,skl->kl',w,Fs)*N
        Fi=np.linalg.inv(F+1e-12*np.eye(len(M)))
        grad=np.einsum('kl,slm,mk->s',Fi@M,Fs,Fi)   # d/dw of -Tr(M F^-1)
        w=w*np.maximum(grad,1e-18)**0.5
        w/=w.sum()
    return w

def pts(e): return 100.0 if e<=1e-3 else 0.0 if e>=0.1 else 100*np.log(0.1/e)/np.log(100.0)

print(f"{'attack':<16}{'N':>7}{'uniform180':>12}{'Aoptimal':>11}{'unif60':>9}{'unif90':>9}"
      f"{'pts_u':>8}{'pts_A':>8}")
print("-"*82)
for nm in ['4q_mixed','4q_ladder','4q_conjugated']:
    tpl=Template.model_validate(json.load(open(f'/Users/aoeuhtns/Documents/q-site-hacks-2026/ionq/dev/attacks/{nm}.json')))
    circ=instantiate(tpl,41,R); labels=labels_of(circ)
    x=np.array([float(g.angle) for g in circ])
    Fs=per_setting_fisher(labels,x); M=hess(labels,x)
    for N in (4000,12000):
        wu=np.ones(180)/180
        eu=obj(Fs,wu,M,N)
        wa=a_optimal(Fs,M,N); ea=obj(Fs,wa,M,N)
        w60=np.zeros(180); w60[:60]=1/60; e60=obj(Fs,w60,M,N)
        w90=np.zeros(180); w90[:90]=1/90; e90=obj(Fs,w90,M,N)
        print(f"{nm:<16}{N:>7}{eu:>12.3e}{ea:>11.3e}{e60:>9.2e}{e90:>9.2e}"
              f"{pts(eu):>8.1f}{pts(ea):>8.1f}",flush=True)
        if N==4000:
            eff=(wa>1e-4).sum()
            print(f"{'':16}{'':7}  A-optimal support: {eff}/180 settings, "
                  f"max weight {wa.max()*180:.1f}x uniform")
