"""Ready-to-run legal baseline using only authenticated oracle count receipts.

Fixed 180-setting public panel. Not a mandatory measurement policy. No hidden
instance/template imports. Dense models are internal initializers; only explicit
rotation-gate patches are submitted and validated under the frozen contract.
"""
from __future__ import annotations
import numpy as np
from duelkit.quantum import G,validate,experiment_index,bitstrings
from .quantum import make_panel,design,Likelihood
from .recovery import Ensemble

def as_gates(patch):
    gates=[]
    for record in patch:
        targets=tuple(i for i,a in enumerate(record['pauli']) if a!='I')
        letters=''.join(record['pauli'][i].lower() for i in targets)
        if len(targets) not in (1,2) or len(set(letters))!=1:
            raise ValueError('Estimated circuit contains an unsupported generator')
        # Equivalent rotations differ by 2*pi only through a global phase.
        angle=float((record['angle']+np.pi)%(2*np.pi)-np.pi)
        gates.append(G('r'+letters,targets,angle))
    return tuple(gates)

def run_defender(client,rules,*,grouped=True,panel_seed=60231):
    if rules.qubits!=4:raise ValueError('This baseline is for the four-qubit ruleset')
    size=min(180,rules.max_settings,rules.block)
    panel=make_panel(4,size,panel_seed)
    ids=[experiment_index(e['prep'],''.join(e['basis']),4) for e in panel]
    kets,bras=design(panel);counts=np.zeros((size,16),dtype=np.int64)
    model=Ensemble(block=grouped,max_gates=rules.patch_max_gates,max_entanglers=rules.patch_max_entanglers)
    for checkpoint in range(rules.checkpoints):
        base,extra=divmod(rules.block,size)
        for j,(index,e) in enumerate(zip(ids,panel)):
            receipt=client.query(index,base+(j<extra),f'c{checkpoint}-s{j}')
            if set(receipt['counts'])!=set(bitstrings(4)):
                raise ValueError('Oracle returned a mismatched outcome register')
            counts[j]+=np.array([receipt['counts'][b] for b in bitstrings(4)],dtype=np.int64)
        fitted=model.fit(Likelihood(kets,bras,counts))
        patch=as_gates(fitted['block'] if grouped else fitted['compact'])
        validate(patch,**rules.validation_kwargs())
        note='grouped/compact likelihood selection' if grouped else 'compact likelihood selection'
        client.submit_patch(patch,note=note)
        client.close_checkpoint()
