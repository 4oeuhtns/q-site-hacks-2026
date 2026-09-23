"""Score every factor candidate at each checkpoint against the true attack (dev only)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame2, harness
sys.path.insert(0, str(harness.DEV_DIR.parent / 'quantum_duel_work/quantum-duel-8q-frame2-0.7/my_solution'))
import f2def
from duelkit.recovery8.quantum import unitary, infidelity
key, seed = sys.argv[1], int(sys.argv[2])
case = [c for c in harness.load_attack_library(frame2.ATTACK8_DIR, seeds=[seed], rules=frame2.RULES)[0] if key in c.label][0]
A = unitary([{'pauli': ''.join(g.name[1].upper() if q in g.targets else 'I' for q in range(8)), 'angle': g.angle} for g in case.circuit], 8)
orig = f2def.Frame2Recovery.checkpoint
def traced(self, client):
    patch, meta = orig(self, client)
    print('stage', client.stage, 'fit loss', round(self.last_fit['loss'], 6), 'x', [round(v, 3) for v in self.last_fit['x']])
    for bic, loss, k, high, tail in self.last_candidates[:10]:
        err = infidelity(A, unitary(high + tail, 8))
        print(f'   bic={bic:12.2f} loss={loss:.6f} k={k} err={err:.2e} high={[(h["pauli"], round(h["angle"], 3)) for h in high]}')
    return patch, meta
f2def.Frame2Recovery.checkpoint = traced
row = harness.run_case(case, frame2.RULES, f2def.run_defender)
print(row['checkpoint_errors'])
