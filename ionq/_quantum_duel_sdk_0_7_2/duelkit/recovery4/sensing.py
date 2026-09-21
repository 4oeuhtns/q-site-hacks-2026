"""Count-only setting selection. Each physical setting includes its analysis operation.
This implementation uses no analysis gates. All retained full bitstrings are available.
"""
from __future__ import annotations
import numpy as np
from .quantum import *

def setting_key(e):
 return (tuple(e['prep']),tuple(e['basis']),tuple((g['pauli'],round(float(g['angle']),12)) for g in e.get('analysis',[])))

def choose_settings(history_panel,counts,estimate_circuit,pool,number,mode='adaptive'):
 used={setting_key(x) for x in history_panel}
 candidates=[x for x in pool if setting_key(x) not in used]
 if len(candidates)<number:raise ValueError('not enough unused candidates')
 if mode=='staged_random':return candidates[:number]
 if mode!='adaptive':raise ValueError('mode')
 if not estimate_circuit:return candidates[:number]
 labels=[g['pauli'] for g in estimate_circuit];x=[g['angle'] for g in estimate_circuit]
 u,J,*_=circuit_and_jac(labels,x,4)
 def information(panel):
  k,b=design(panel);a=np.einsum('sij,sj->si',b,k@u.T,optimize=True)
  da=np.einsum('sij,kja,sa->sik',b,J,k,optimize=True)
  dp=2*np.real(a.conj()[:,:,None]*da);p=np.maximum(abs(a)**2,1e-7)
  return np.einsum('sok,sol,so->skl',dp,dp,1/p,optimize=True)
 old=information(history_panel);new=information(candidates);weights=np.sum(counts,axis=1)
 F=np.einsum('s,sij->ij',weights,old,optimize=True)+np.eye(len(labels))*.05
 selected=[];available=np.ones(len(candidates),bool);newshots=4000/(len(history_panel)+number)
 for j in range(number):
  inv=np.linalg.inv(F);utility=np.einsum('ij,sji->s',inv,new,optimize=True);utility[~available]=-np.inf
  # Every fifth setting is uninformed exploration, avoiding pure model lock-in.
  idx=int(np.flatnonzero(available)[0]) if j%5==4 else int(np.argmax(utility))
  selected.append(candidates[idx]);available[idx]=False;F+=newshots*new[idx]
 return selected
