"""Generate the public-architecture library shipped with the defender (tlib.json).

Every entry is something a team can produce by calling a public SDK helper or
copying a notebook cell without editing it.
"""
import json, sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_quantum_duel_sdk_0_7_2"))
from qduel_sdk.profiles import (profile_rules, open_example, open_practice_bank, open_multilayer_example,
                                frame_example, merged_frame_example)
from qduel_sdk.rules import OPEN8_RULESET, FRAME2_RULESET, FRAME4_RULESET
from qduel_sdk.contracts import Template

R = profile_rules(OPEN8_RULESET)
lib = {}

def add(key, t):
    T = Template.model_validate(t)
    gates = []
    for g in T.gates:
        if g.parameter is not None:
            p = T.parameters[g.parameter]
            lo, hi = sorted((g.scale * p.low + g.offset, g.scale * p.high + g.offset))
        elif g.low is not None:
            lo, hi = g.low, g.high
        else:
            continue
        gates.append([g.name, list(g.targets), round(lo, 4), round(hi, 4)])
    lib[key] = gates

for k, t in open_practice_bank(R).items():
    add(f"bank:{k}", t)
for s in range(0, 128):
    for g, e in [(36, 12), (18, 6), (72, 24)]:
        add(f"open_example:{s}:{g}:{e}", open_example(R, s, gates=g, entanglers=e))
    add(f"multilayer:{s}", open_multilayer_example(R, s))
add("notebook:open_demo", {"name": "Repeated-pair open example", "parameters": {"a": {"low": 0.45, "high": 0.65}},
    "gates": [
        {"name": "rzz", "targets": [0, 7], "parameter": "a"},
        {"name": "rx", "targets": [0], "low": 0.8, "high": 1.0},
        {"name": "ryy", "targets": [2, 7], "low": -0.8, "high": -0.5},
        {"name": "rzz", "targets": [0, 7], "parameter": "a", "scale": -1.0},
        {"name": "rz", "targets": [3], "low": 0.9, "high": 1.1}]})
for s in (23, 29, 31, 37, 41, 73, 97):
    add(f"frame2:{s}", frame_example(profile_rules(FRAME2_RULESET), s))
    add(f"frame4:{s}", frame_example(profile_rules(FRAME4_RULESET), s))
add("merged:127", merged_frame_example(profile_rules(FRAME2_RULESET), 127))
out = Path(__file__).resolve().parents[2] / "quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/tlib.json"
out.write_text(json.dumps(lib, separators=(",", ":")))
print(len(lib), "entries", out.stat().st_size // 1024, "KiB")
