"""Morning read-out for queue_overnight.sh (run from anywhere)."""
import collections
import json
import re
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "data"
rows = [json.loads(line) for line in open(D / "results.jsonl")]
by = collections.defaultdict(list)
for r in rows:
    by[r.get("tag")].append(r)


def name(r):
    return r.get("template_name") or json.dumps(r.get("spec"))


def health(tag):
    rs = by.get(tag, [])
    if not rs:
        print(f"\n=== {tag}: no rows yet")
        return rs
    pts = [r["points"] for r in rs if r.get("points") is not None]
    st = collections.Counter(r["status"] for r in rs)
    print(f"\n=== {tag}: n={len(rs)} mean={np.mean(pts):.1f} zeros={sum(p == 0 for p in pts)} status={dict(st)}")
    print(f"    cpu max {max(r.get('cpu', 0) for r in rs):.0f}s, wall max {max(r.get('wall', 0) for r in rs):.0f}s")
    for r in rs:
        if r["status"] not in ("PASSED",):
            print(f"    !! {r['status']:<14} {name(r)}  {r.get('error', '')[:150]}")
        elif r.get("cpu", 0) > 620:
            print(f"    !! CPU {r['cpu']:.0f}s  {name(r)}")
    return rs


# 1. fuzz
rs = health("fuzz_v11")
if rs:
    print("    by size (gates):  n  mean  zeros")
    grp = collections.defaultdict(list)
    for r in rs:
        if r.get("points") is not None:
            grp[min(r.get("gates", 0) // 12 * 12, 72)].append(r["points"])
    for g in sorted(grp):
        print(f"    {g:>3}-{g + 11:<3}          {len(grp[g]):>3} {np.mean(grp[g]):>5.1f} {sum(p == 0 for p in grp[g]):>4}")
    print("    zeros at <= 24 gates (should be recoverable):")
    for r in sorted(rs, key=lambda r: r.get("gates", 0)):
        if r.get("points") == 0 and r.get("gates", 99) <= 24:
            print(f"      {name(r)}  cpu={r.get('cpu', 0):.0f}")
    print("    edge cases:")
    for r in rs:
        if name(r).startswith("edge"):
            print(f"      {r.get('points', 0):>6.1f}  {name(r)}  cp={[round(x, 1) for x in r.get('cp_points', [])]}")

# 2. edits
rs = health("edits2_v11")
if rs:
    tab = collections.defaultdict(dict)
    for r in rs:
        m = re.match(r"edit (\S+) (\S+) v(\d)", name(r))
        if m and r.get("points") is not None:
            tab[m[2]][(m[1], m[3])] = r["points"]
    bases = ["mixed", "ex23_36", "multilayer", "ex23_72", "frame"]
    print("    " + f"{'edit':<11}" + "".join(f"{b:>15}" for b in bases))
    for op in tab:
        cells = []
        for b in bases:
            v = [tab[op].get((b, x)) for x in "12"]
            cells.append("/".join("-" if x is None else f"{x:.0f}" for x in v))
        print("    " + f"{op:<11}" + "".join(f"{c:>15}" for c in cells))

# 3. cap 400 vs cap 600 (paired on spec + draw)
rs = health("v11cap400")
if rs:
    base = {(json.dumps(r["spec"], sort_keys=True), r["draw"]): r for r in by["v11"]}
    diffs = []
    for r in rs:
        b = base.get((json.dumps(r["spec"], sort_keys=True), r["draw"]))
        if b and r.get("points") is not None and b.get("points") is not None:
            diffs.append((r["points"] - b["points"], name(r), r["draw"], b["points"], r["points"], b.get("cpu", 0), r.get("cpu", 0)))
    if diffs:
        print(f"    paired n={len(diffs)}: cap600 mean {np.mean([d[3] for d in diffs]):.1f} -> cap400 {np.mean([d[4] for d in diffs]):.1f}")
        for d in sorted(diffs):
            if abs(d[0]) >= 5:
                print(f"      {d[0]:+6.1f}  {d[1]} @{d[2]}  {d[3]:.1f} -> {d[4]:.1f}  cpu {d[5]:.0f} -> {d[6]:.0f}")

# 4. offline pursuit engines
f = D / "frontier_overnight.log"
if f.exists():
    res = collections.defaultdict(list)
    for line in open(f):
        try:
            j = json.loads(line)
        except ValueError:
            continue
        res[(j["algo"], j["g"])].append(j["pts"])
    print("\n=== frontier_overnight (96k shots, 400 s, seeds 24-29)")
    for k in sorted(res):
        v = res[k]
        print(f"    {k[0]} {k[1]}g: n={len(v)} mean={np.mean(v):5.1f}  solved(>=90)={sum(x >= 90 for x in v)}  {v}")
