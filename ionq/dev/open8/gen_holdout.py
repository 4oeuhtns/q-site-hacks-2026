"""Held-out attack set for v11 vs v12 (run from ionq/).

    python3 dev/open8/gen_holdout.py

v12 was designed from the edits2/fuzz/v11-set failures, so its scores there are
in-sample. This set was generated after that design and is not looked at before
the paired run: new edit instances (variant 3), edit types and sizes never used
(qubit relabelling, angle scaling, axis2, retarget3, delete8, combo2), one base
not used in design (open_example seed 23 at 18 g), and 40 new fuzz seeds.

Writes dev/open8/specs/holdout/*.json and specs_holdout.json.
"""
import copy
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_overnight as go  # noqa: E402  (reuses its gate/edit/fuzz/legal helpers)

OUT = HERE / "specs/holdout"
REL = "dev/open8/specs/holdout"
PI = go.PI


def edit(gates, op, k, rng):
    gs = copy.deepcopy(gates)
    if op == "permute":          # relabel every qubit
        perm = [int(x) for x in rng.permutation(8)]
        for g in gs:
            g["targets"] = sorted(perm[t] for t in g["targets"])
        return gs
    if op == "scale":            # every angle range scaled by k/10
        out = []
        for g in gs:
            c, w = (g["low"] + g["high"]) / 2 * k / 10, g["high"] - g["low"]
            out.append(go.gate(g["name"][1], g["targets"], c, w))
        return out
    if op == "combo2":
        gs = go.edit(gs, "axis", 1, rng)
        gs = go.edit(gs, "retarget", 1, rng)
        return go.edit(gs, "delete", 2, rng)
    return go.edit(gs, op, k, rng)


OPS = [("axis", 2), ("retarget", 3), ("delete", 8), ("insert", 2), ("shift", 2), ("widen", 5),
       ("swap", 3), ("combo2", 0), ("permute", 0), ("scale", 8)]


def edits():
    bank = go.open_practice_bank(go.R)
    bases = {"mixed": bank["mixed"], "multilayer": bank["multilayer"],
             "ex23_18": go.open_example(go.R, 23, gates=18, entanglers=6),
             "ex23_36": go.open_example(go.R, 23, gates=36, entanglers=12),
             "ex23_72": go.open_example(go.R, 23, gates=72, entanglers=24)}
    out = {}
    for b, t in bases.items():
        for op, k in OPS:
            rng = np.random.default_rng([zlib.crc32(b.encode()), zlib.crc32(op.encode()), k, 3])
            gs = edit(t["gates"], op, k, rng)
            if len(gs) > 72 or sum(len(g["targets"]) == 2 for g in gs) > 24:
                continue
            out[f"hold {b} {op}{k}"] = gs
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {**edits(), **{f"hold {t['name']}": t["gates"] for t in map(go.fuzz, range(100, 140))}}
    specs, bad = [], 0
    for name, gates in cases.items():
        try:
            t = go.legal(name, gates)
        except Exception as exc:
            bad += 1
            print("ILLEGAL", name, type(exc).__name__, str(exc)[:120])
            continue
        fn = re.sub(r"[^A-Za-z0-9_-]+", "_", name) + ".json"
        (OUT / fn).write_text(json.dumps(t))
        specs.append(dict(family="file", path=f"{REL}/{fn}"))
    assert len({s["path"] for s in specs}) == len(specs), "file name collision"
    (HERE / "specs_holdout.json").write_text(json.dumps(specs, indent=0))
    print(f"holdout: {len(specs)} specs, {bad} dropped")
