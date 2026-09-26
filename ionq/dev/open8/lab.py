"""Open-final (8q) lab: attack families, encounter runner, JSONL logging.

DEVELOPMENT ONLY. Knows the attack; never import from a submitted main.py.

    python3 dev/open8/lab.py --defender PATH --family frame_k --tag r1 -j 8

Every encounter appends one row to dev/open8/data/results.jsonl, so every run is data.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
IONQ = HERE.parents[1]
sys.path.insert(0, str(IONQ / "_quantum_duel_sdk_0_7_2"))

from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.local import LocalSession  # noqa: E402
from qduel_sdk.profiles import (profile_rules, open_example, open_practice_bank,  # noqa: E402
                                frame_example, merged_frame_example)
from qduel_sdk.rules import OPEN8_RULESET, FRAME2_RULESET, FRAME4_RULESET  # noqa: E402
from qduel_sdk.templates import instantiate  # noqa: E402

RULES = profile_rules(OPEN8_RULESET)
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
Q = math.pi / 2
N = 8

# ------------------------------------------------------------------ families
# Each family(seed) -> template dict. The *architecture* seed is the family seed;
# the draw seed (angles) is separate, exactly as on the server.


def _pairs_matching(rng):
    perm = rng.permutation(N)
    return [sorted(map(int, perm[j:j + 2])) for j in range(0, N, 2)]


def frame(seed, k=2, ins_lo=0.3, ins_hi=0.9, width=0.0, layers=6, pair_width=0.0,
          pair_ins=0):
    """Cup-2-style frame: `layers` x (8 locals + 4 matching pairs).

    k local insertions with |angle| in [ins_lo, ins_hi]; other slots at +-pi/2,
    optionally ranged by +-width (near-Clifford). pair_ins pair slots get
    non-Clifford ranges too.
    """
    rng = np.random.default_rng(seed)
    local_slots = [(l, q) for l in range(layers) for q in range(N)]
    ins = set(map(int, rng.choice(len(local_slots), k, replace=False))) if k else set()
    pair_slots = [(l, j) for l in range(layers) for j in range(4)]
    pins = set(map(int, rng.choice(len(pair_slots), pair_ins, replace=False))) if pair_ins else set()
    gates, si, pi_ = [], 0, 0
    for l in range(layers):
        for q in rng.permutation(N):
            ax = str(rng.choice(list("xyz")))
            s = int(rng.choice([-1, 1]))
            if si in ins:
                lo, hi = sorted((s * ins_lo, s * ins_hi))
            else:
                lo, hi = s * Q - width, s * Q + width
            gates.append(dict(name="r" + ax, targets=[int(q)], low=lo, high=hi))
            si += 1
        for pr in _pairs_matching(rng):
            ax = str(rng.choice(list("xyz")))
            s = int(rng.choice([-1, 1]))
            if pi_ in pins:
                lo, hi = sorted((s * ins_lo, s * ins_hi))
            else:
                lo, hi = s * Q - pair_width, s * Q + pair_width
            gates.append(dict(name="r" + ax * 2, targets=pr, low=lo, high=hi))
            pi_ += 1
    return dict(name=f"frame k{k} w{width} pw{pair_width} pins{pair_ins} / {seed}", gates=gates)


def generic(seed, gates=72, ents=24, lo=0.5, hi=2.0, width=0.15, pairs="random", bands=None, qubits=None):
    """Continuous-angle random architecture, like open_example but tunable."""
    rng = np.random.default_rng(seed)
    slots = set(map(int, rng.choice(gates, ents, replace=False)))
    allpairs = [(a, b) for a in range(N) for b in range(a + 1, N)]
    out, pc = [], 0
    for i in range(gates):
        ax = str(rng.choice(list("xyz")))
        if i in slots:
            if pairs == "brick":
                off = (pc // 4) % 2
                a = (2 * (pc % 4) + off) % N
                t = sorted([a, (a + 1) % N])
            elif pairs == "all":
                t = list(allpairs[int(rng.integers(len(allpairs)))])
            elif qubits:
                t = sorted(map(int, rng.choice(qubits, 2, replace=False)))
            else:
                t = sorted(map(int, rng.choice(N, 2, replace=False)))
            pc += 1
            name = "r" + ax * 2
        else:
            t = [int(rng.choice(qubits))] if qubits else [int(rng.integers(N))]
            name = "r" + ax
        if bands:
            b = bands[int(rng.integers(len(bands)))]
            c = float(rng.uniform(*b)) * int(rng.choice([-1, 1]))
        else:
            c = float(rng.uniform(lo, hi)) * int(rng.choice([-1, 1]))
        l, h = max(-math.pi, c - width), min(math.pi, c + width)
        out.append(dict(name=name, targets=t, low=l, high=h))
    tag = f" q{len(qubits)}" if qubits else ""
    return dict(name=f"gen g{gates} e{ents} {'band' if bands else f'{lo}-{hi}'} {pairs}{tag} /{seed}", gates=out)


def sparse(seed, gates=6, ents=2):
    return generic(seed, gates=gates, ents=ents, lo=0.4, hi=2.6, width=0.15)


def pub(name):
    bank = open_practice_bank(RULES)
    return bank[name]


def cup2_own(which):
    zp = IONQ / "quantum_duel_work/quantum-duel-8q-frame2-0.7/submission.zip"
    import zipfile
    d = json.loads(zipfile.ZipFile(zp).read("attacks.json"))
    ts = d if isinstance(d, list) else d.get("templates", d.get("attacks"))
    return ts[which]


def spec_to_template(spec):
    """spec: dict(family=..., **kwargs) -> template dict."""
    spec = dict(spec)
    fam = spec.pop("family")
    if fam == "frame":
        return frame(**spec)
    if fam == "generic":
        return generic(**spec)
    if fam == "sparse":
        return sparse(**spec)
    if fam == "open_example":
        return open_example(RULES, spec.get("seed", 23), gates=spec.get("gates", 36),
                            entanglers=spec.get("entanglers", 12))
    if fam == "pub":
        return pub(spec["name"])
    if fam == "frame_example":
        return frame_example(profile_rules(FRAME2_RULESET if spec.get("t", 2) == 2 else FRAME4_RULESET),
                             spec.get("seed", 23))
    if fam == "merged":
        return merged_frame_example(profile_rules(FRAME2_RULESET), spec.get("seed", 127))
    if fam == "cup2_own":
        return cup2_own(spec["which"])
    if fam == "file":
        return json.loads(Path(spec["path"]).read_text())
    raise ValueError(fam)


# ------------------------------------------------------------------ running

def load_defender(path):
    if path in (None, "stock"):
        from duelkit.recovery8.adapter import run_defender
        return lambda c, r: run_defender(c, r)
    if path == "starter":
        from duelkit.recovery8.open_starter import run_defender
        return lambda c, r: run_defender(c, r)
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(f"def_{abs(hash(str(p)))}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.run


_DEF = {}
# Rated runs reach the oracle over HTTP (~31 ms per request, SDK client.py). With
# LAB_LATENCY_MS set, every client call sleeps first: wall time without CPU time.
LATENCY = float(os.environ.get("LAB_LATENCY_MS", "0")) / 1000.0
_NETWORK = {"query", "query_batch", "query_experiment", "status", "submit_patch", "close_checkpoint", "finish"}


class LatentClient:
    def __init__(self, client):
        self._client = client

    def __getattr__(self, name):
        attr = getattr(self._client, name)
        if name not in _NETWORK or not LATENCY:
            return attr

        def call(*a, **k):
            time.sleep(LATENCY)
            return attr(*a, **k)
        return call


def run_one(job):
    spec, draw, defender = job
    t0w, t0c = time.time(), time.process_time()
    row = dict(spec=spec, draw=draw, defender=defender)
    try:
        tdict = spec_to_template(spec)
        row["template_name"] = tdict.get("name")
        t = Template.model_validate(tdict)
        circ = instantiate(t, int(draw), RULES)
        row["gates"] = len(circ)
        row["ents"] = sum(len(g.targets) == 2 for g in circ)
    except Exception as exc:
        row.update(status="ADMISSION_FAIL", error=f"{type(exc).__name__}: {exc}"[:300], points=None)
        return row
    if defender not in _DEF:
        _DEF[defender] = load_defender(defender)
    sess = LocalSession(circ, RULES, seed=int(draw) + 100_000)
    try:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            t_run = time.time()
            _DEF[defender](LatentClient(sess.client()), RULES)
            row["run_wall"] = time.time() - t_run
        sess.client().finish()
        res = sess.result()
        cps = res["checkpoint_scores"]
        row.update(status="PASSED", points=res["recovery_points"],
                   cp_points=[c["recovery_points"] for c in cps],
                   cp_eps=[c["process_infidelity"] for c in cps],
                   cp_gates=[len(c.get("patch") or []) for c in cps],
                   cp_ents=[sum(len(g["targets"]) == 2 for g in c.get("patch") or []) for c in cps],
                   shots=res["spent_shots"], settings=res["distinct_settings"],
                   log=buf.getvalue()[-6000:])
    except BaseException as exc:
        row.update(status="FAILED", points=0.0, error=f"{type(exc).__name__}: {exc}"[:300],
                   tb=traceback.format_exc()[-800:])
    row["wall"] = time.time() - t0w
    row["cpu"] = time.process_time() - t0c
    return row


def run_jobs(jobs, n_proc, tag, out=DATA / "results.jsonl", verbose=True):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = []
    with ProcessPoolExecutor(max_workers=n_proc) as pool, open(out, "a") as fh:
        futs = [pool.submit(run_one, j) for j in jobs]
        for f in as_completed(futs):
            try:
                r = f.result()
            except BaseException as exc:
                r = dict(status="WORKER_CRASH", error=str(exc)[:300], points=0.0)
            r["tag"] = tag
            r["ts"] = time.time()
            fh.write(json.dumps(r, default=str) + "\n")
            fh.flush()
            rows.append(r)
            if verbose:
                pts = r.get("points")
                print(f"{r.get('status'):<15} {pts if pts is None else round(pts,1)!s:>6} "
                      f"cp={[round(x,1) for x in r.get('cp_points',[])]} "
                      f"cpu={r.get('cpu',0):.0f}s  {r.get('template_name','?')} @{r.get('draw')} "
                      f"{r.get('error','')[:80]}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--defender", default="stock")
    ap.add_argument("--specs", required=True, help="JSON list of spec dicts, or @file")
    ap.add_argument("--draws", type=int, nargs="+", default=[1])
    ap.add_argument("--tag", default="adhoc")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    specs = json.loads(Path(a.specs[1:]).read_text() if a.specs.startswith("@") else a.specs)
    jobs = [(s, d, a.defender) for s in specs for d in a.draws]
    rows = run_jobs(jobs, a.j, a.tag)
    ok = [r["points"] for r in rows if r.get("points") is not None]
    print(f"\nn={len(rows)} mean={np.mean(ok) if ok else float('nan'):.1f} "
          f"zeros={sum(p == 0 for p in ok)} fails={sum(r['status'] != 'PASSED' for r in rows)}")
