"""Run f2def on one public case with learner meta printed per checkpoint."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame2, harness
sys.path.insert(0, str(harness.DEV_DIR.parent / 'quantum_duel_work/quantum-duel-8q-frame2-0.7/my_solution'))
import f2def
case = [c for c in harness.load_attack_library(frame2.ATTACK8_DIR, seeds=[int(sys.argv[2])], rules=frame2.RULES)[0] if sys.argv[1] in c.label][0]
orig = f2def.Frame2Recovery.checkpoint
def traced(self, client):
    patch, meta = orig(self, client)
    print('stage', client.stage, {k: v for k, v in meta.items() if k not in ('fit', 'compile_attempts')})
    return patch, meta
f2def.Frame2Recovery.checkpoint = traced
row = harness.run_case(case, frame2.RULES, f2def.run_defender)
print(row['checkpoint_points'], row['checkpoint_errors'])
