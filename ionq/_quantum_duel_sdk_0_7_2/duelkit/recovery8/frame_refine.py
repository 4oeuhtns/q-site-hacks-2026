"""Budget-gated extra Clifford elimination search, followed by exact-equivalence check.
Consumes only the learner's proposed Pauli network, never the hidden attack.
"""
import numpy as np
from .pauli_compile import compile_network as baseline_compile, clifford_absorb, network
from .quantum import unitary

def compile_refined(paulis,tail,n,gate_cap=108,ent_cap=36,trials=64,drop=0.):
    gs,meta=baseline_compile(paulis,tail,n,gate_cap,ent_cap,trials,drop)
    meta=dict(meta,compiler='budget_gated_tail_search_v1',extra_trials=0)
    if meta['legal']:
        return gs,meta
    non,rows=clifford_absorb(paulis,tail,n,drop)
    incumbent=gs
    for seed in range(128):
        proposal=network(non,rows,n,seed,tail_trials=32)
        e=sum(sum(a!='I' for a in g['pauli'])==2 for g in proposal)
        key=(e>ent_cap or len(proposal)>gate_cap,e,len(proposal))
        old=(meta['entanglers']>ent_cap or meta['gates']>gate_cap,meta['entanglers'],meta['gates'])
        if key<old:
            gs=proposal;meta.update(entanglers=e,gates=len(gs),legal=not key[0])
        meta['extra_trials']=seed+1
        if meta['legal']:break
    # Compiler equivalence is against the previous representation, not the private target.
    a=unitary(incumbent,n);p=unitary(gs,n)
    e=max(0.,1-abs(np.vdot(a,p))**2/len(a)**2)
    if e>1e-10:raise RuntimeError('compiler changed process')
    meta['equivalence_error']=float(e)
    return gs,meta
