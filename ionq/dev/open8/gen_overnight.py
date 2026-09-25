"""Generate the overnight attack sets (run from ionq/).

    python3 dev/open8/gen_overnight.py

fuzz   -- random legal attacks over the whole attack grammar plus hand-picked edge
          cases. Purpose: find crashes, over-cap CPU and surprise zeros in v11.
edits2 -- public templates edited the way a team might (axes, targets, deletions,
          insertions, angle shifts/widening, reordering). Purpose: map where the
          template screen and seeded repair stop working.

Writes dev/open8/specs/overnight/*.json and the lists specs_fuzz.json, specs_edits2.json.
"""
import copy
import json
import math
import re
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402  (sets sys.path for the SDK)
from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.templates import instantiate  # noqa: E402
from qduel_sdk.profiles import open_example, open_practice_bank  # noqa: E402
from duelkit.quantum import validate  # noqa: E402

PI = math.pi
R = lab.RULES
OUT = HERE / "specs/overnight"
REL = "dev/open8/specs/overnight"


def clip(c):
    return float(min(PI, max(-PI, c)))


def gate(axis, targets, c, w=0.3):
    c = clip(c)
    lo, hi = clip(c - w / 2), clip(c + w / 2)
    return dict(name="r" + axis * len(targets), targets=sorted(int(t) for t in targets), low=lo, high=hi)


def centre(rng, mode):
    s = float(rng.choice([-1, 1]))
    if mode == "uniform":
        return float(rng.uniform(-PI, PI))
    if mode == "bands":
        return s * float(rng.uniform(0.5, 2.0))
    if mode == "small":
        return s * float(rng.uniform(0.1, 0.4))
    if mode == "clifford":
        return float(rng.choice([PI / 2, -PI / 2, PI, -PI]))
    if mode == "nearcliff":
        return clip(float(rng.choice([PI / 2, -PI / 2, PI])) + float(rng.normal(0, 0.1)))
    if mode == "boundary":
        return s * float(rng.uniform(2.8, PI))
    raise ValueError(mode)


def pair(rng, mode, Q, fixed):
    if mode == "random":
        return [int(x) for x in rng.choice(Q, 2, replace=False)]
    if mode == "line":
        i = int(rng.integers(len(Q) - 1))
        return [Q[i], Q[i + 1]]
    if mode == "star":
        return [Q[0], int(rng.choice(Q[1:]))]
    if mode == "few":
        return list(fixed[int(rng.integers(len(fixed)))])
    raise ValueError(mode)


# ------------------------------------------------------------------ fuzz

def fuzz(i):
    rng = np.random.default_rng(90_000 + i)
    bucket = int(rng.choice(4, p=[0.3, 0.35, 0.2, 0.15]))
    G = int(rng.integers(*[(1, 9), (9, 25), (25, 49), (49, 73)][bucket]))
    nq = int(rng.choice([8, 8, 8, 2, 3, 5]))
    Q = sorted(int(q) for q in rng.choice(8, nq, replace=False))
    frac = float(rng.choice([0.0, 0.15, 1 / 3, 0.5, 1.0]))
    E = min(24, G, int(rng.binomial(G, frac)))
    pm = str(rng.choice(["random", "line", "star", "few"]))
    am = str(rng.choice(["uniform", "bands", "bands", "small", "clifford", "nearcliff", "boundary"]))
    w = 0.0 if am == "clifford" else float(rng.choice([0.0, 0.1, 0.3, 1.0]))
    fixed = [pair(rng, "random", Q, None) for _ in range(2)]
    ent_at = set(int(x) for x in rng.choice(G, E, replace=False))
    gates = []
    for k in range(G):
        tg = pair(rng, pm, Q, fixed) if k in ent_at else [int(rng.choice(Q))]
        gates.append(gate(str(rng.choice(list("xyz"))), tg, centre(rng, am), w))
    return dict(name=f"fuzz{i:03d} G{G} E{E} q{nq} {pm} {am} w{w:g}", gates=gates)


def edge_cases():
    rng = np.random.default_rng(4242)
    cases = {}

    def rnd(G, E, am="uniform", w=0.3, Q=tuple(range(8)), pm="random"):
        Q = list(Q)
        fixed = [pair(rng, "random", Q, None) for _ in range(2)] if len(Q) > 1 else []
        ent_at = set(int(x) for x in rng.choice(G, E, replace=False))
        return [gate(str(rng.choice(list("xyz"))), pair(rng, pm, Q, fixed) if k in ent_at else [int(rng.choice(Q))],
                     centre(rng, am), w) for k in range(G)]

    cases["one_rx_pi"] = [gate("x", [0], PI, 0)]
    cases["one_rzz_minus_pi"] = [gate("z", [0, 7], -PI, 0)]
    cases["locals_only_72"] = rnd(72, 0)
    cases["one_pair_24ent"] = rnd(72, 24, Q=(3, 4))
    cases["ents_only_24"] = rnd(24, 24, am="bands")
    cases["single_qubit_72"] = rnd(72, 0, am="bands", Q=(0,))
    cases["two_qubit_72"] = rnd(72, 24, am="bands", Q=(0, 1))
    cases["repeat_rx_72"] = [gate("x", [2], 1.3, 0.3) for _ in range(72)]
    cases["clifford_generic_72"] = rnd(72, 24, am="clifford", w=0)
    cl = rnd(72, 24, am="clifford", w=0)
    for k in rng.choice([j for j, g in enumerate(cl) if len(g["targets"]) == 1], 4, replace=False):
        g = cl[int(k)]
        cl[int(k)] = gate(g["name"][1], g["targets"], 0.6, 0.6)
    cases["clifford_generic_4ins"] = cl
    cases["near_identity_72"] = rnd(72, 24, am="small", w=0.1)
    cases["boundary_72"] = rnd(72, 24, am="boundary", w=0.3)
    cases["star_48"] = rnd(48, 16, am="bands", pm="star")
    cases["diag_zz_z_72"] = [gate("z", g["targets"], g["low"] + 0.15, 0.3) for g in rnd(72, 24, am="bands")]
    cases["diag_xx_x_36"] = [gate("x", g["targets"], g["low"] + 0.15, 0.3) for g in rnd(36, 12, am="bands")]
    sand = [gate(str(rng.choice(list("xyz"))), [q], centre(rng, "bands")) for q in range(8)]
    sand += [gate("z", [q, q + 1], centre(rng, "bands")) for q in range(0, 7)]
    sand += [gate(str(rng.choice(list("xyz"))), [q], centre(rng, "bands")) for q in range(8)]
    cases["sandwich_23"] = sand
    cases["full_range_w2pi_24"] = [dict(g, low=-PI, high=PI) for g in rnd(24, 8)]
    return {f"edge {k}": v for k, v in cases.items()}


# ------------------------------------------------------------------ edits

def edit(base_gates, op, k, rng):
    gs = copy.deepcopy(base_gates)
    def cen(g):
        return (g["low"] + g["high"]) / 2

    def wid(g):
        return g["high"] - g["low"]

    if op == "axis":
        for i in rng.choice(len(gs), k, replace=False):
            g = gs[int(i)]
            a = str(rng.choice([x for x in "xyz" if x != g["name"][1]]))
            g["name"] = "r" + a * len(g["targets"])
    elif op == "retarget":
        for i in rng.choice(len(gs), k, replace=False):
            g = gs[int(i)]
            old = sorted(g["targets"])
            while sorted(g["targets"]) == old:
                g["targets"] = sorted(int(x) for x in rng.choice(8, len(old), replace=False))
    elif op == "delete":
        drop = set(int(i) for i in rng.choice(len(gs), k, replace=False))
        gs = [g for i, g in enumerate(gs) if i not in drop]
    elif op == "insert":
        ents = sum(len(g["targets"]) == 2 for g in gs)
        for _ in range(k):
            two = ents < 24 and rng.random() < 1 / 3
            ents += two
            tg = rng.choice(8, 2 if two else 1, replace=False)
            gs.insert(int(rng.integers(len(gs) + 1)), gate(str(rng.choice(list("xyz"))), tg, centre(rng, "bands")))
    elif op == "shift":
        gs = [gate(g["name"][1], g["targets"], (cen(g) + k / 10 + PI) % (2 * PI) - PI, wid(g)) for g in gs]
    elif op == "widen":
        gs = [gate(g["name"][1], g["targets"], cen(g), k / 10) for g in gs]
    elif op == "negate":
        gs = [gate(g["name"][1], g["targets"], -cen(g), wid(g)) for g in gs]
    elif op == "reverse":
        gs = gs[::-1]
    elif op == "swap":
        for i in rng.choice(len(gs) - 1, k, replace=False):
            i = int(i)
            gs[i], gs[i + 1] = gs[i + 1], gs[i]
    elif op == "combo":
        gs = edit(gs, "axis", 3, rng)
        gs = edit(gs, "retarget", 2, rng)
        gs = edit(gs, "delete", 4, rng)
    else:
        raise ValueError(op)
    return gs


CONT_OPS = [("axis", 1), ("axis", 3), ("axis", 6), ("retarget", 1), ("retarget", 2), ("retarget", 4),
            ("delete", 4), ("delete", 12), ("insert", 4), ("shift", 4), ("widen", 10), ("negate", 0),
            ("reverse", 0), ("swap", 6), ("combo", 0)]
FRAME_OPS = [("axis", 3), ("retarget", 2), ("delete", 4), ("reverse", 0), ("swap", 6)]


def edits():
    bank = open_practice_bank(R)
    bases = {"mixed": bank["mixed"], "multilayer": bank["multilayer"], "frame": bank["frame"],
             "ex23_36": open_example(R, 23, gates=36, entanglers=12),
             "ex23_72": open_example(R, 23, gates=72, entanglers=24)}
    out = {}
    for b, t in bases.items():
        for op, k in (FRAME_OPS if b == "frame" else CONT_OPS):
            for v in (1, 2):
                rng = np.random.default_rng([zlib.crc32(b.encode()), zlib.crc32(op.encode()), k, v])
                gs = edit(t["gates"], op, k, rng)
                if len(gs) > 72 or sum(len(g["targets"]) == 2 for g in gs) > 24:
                    continue
                out[f"edit {b} {op}{k} v{v}"] = gs
    return out


def legal(name, gates):
    t = dict(name=name[:64], gates=gates)
    T = Template.model_validate(t)
    for d in (1, 2):
        validate(instantiate(T, d, R), **R.validation_kwargs("attack"))
    return t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    sets = {"fuzz": {**edge_cases(), **{t["name"]: t["gates"] for t in map(fuzz, range(100))}},
            "edits2": edits()}
    for key, cases in sets.items():
        specs, bad = [], 0
        for name, gates in cases.items():
            try:
                t = legal(name, gates)
            except Exception as exc:
                bad += 1
                print("ILLEGAL", name, type(exc).__name__, str(exc)[:120])
                continue
            fn = re.sub(r"[^A-Za-z0-9_-]+", "_", name) + ".json"
            (OUT / fn).write_text(json.dumps(t))
            specs.append(dict(family="file", path=f"{REL}/{fn}"))
        assert len({s["path"] for s in specs}) == len(specs), "file name collision"
        (HERE / f"specs_{key}.json").write_text(json.dumps(specs, indent=0))
        print(f"{key}: {len(specs)} specs, {bad} dropped")
