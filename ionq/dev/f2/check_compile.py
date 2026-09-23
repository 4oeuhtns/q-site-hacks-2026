"""Compile exact inverses of frame2 draws with f2def's search; check equivalence and cost."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_quantum_duel_sdk_0_7_2'))
sys.path.insert(0, str(ROOT / 'quantum_duel_work/quantum-duel-8q-frame2-0.7/my_solution'))
import numpy as np
from qduel_sdk.rules import FRAME2_RULESET
from qduel_sdk.profiles import profile_rules, practice_bank
from qduel_sdk.contracts import Template
from qduel_sdk.templates import instantiate
from duelkit.recovery8.quantum import unitary, infidelity
from duelkit.recovery8.pauli_compile import clifford_absorb
import f2def

rules = profile_rules(FRAME2_RULESET)

def as_pauli(circuit):
    out = []
    for g in circuit:
        axis = g.name[1].upper()
        out.append({'pauli': ''.join(axis if q in g.targets else 'I' for q in range(8)), 'angle': float(g.angle)})
    return out

for key, t in practice_bank(rules).items():
    attack = as_pauli(instantiate(Template(**t), 41, rules))
    inv = f2def.inverse(attack)
    non, rows = clifford_absorb(inv, [], 8, 1e-8)
    t0 = time.process_time()
    best, rounds = f2def.search([(non, rows, 0.0)], 8, 108, 36, time.process_time() + 10, np.random.default_rng(1))
    gs = best[1]
    err = infidelity(unitary(attack, 8), unitary(gs, 8))
    print(f'{key:8s} rotations={len(non)} ent={f2def.entanglers(gs)} gates={len(gs)} rounds={rounds} '
          f'cpu={time.process_time()-t0:.1f}s inverse_err={err:.2e}')
