"""Aggregate dev/open8/data/results.jsonl: mean/zero-rate per (tag, family-key)."""
import json, sys, collections
from pathlib import Path
import numpy as np
rows = [json.loads(l) for l in open(Path(__file__).parent / "data/results.jsonl")]
tags = sys.argv[1:] or sorted({r.get("tag") for r in rows})
def key(s):
    s = dict(s); fam = s.pop("family"); s.pop("seed", None)
    return fam + " " + " ".join(f"{k}={v}" for k, v in sorted(s.items()))
for tag in tags:
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("tag") == tag and r.get("points") is not None:
            g[key(r["spec"])].append(r)
    if not g: continue
    print(f"\n=== {tag}")
    print(f"{'family':<48}{'n':>3}{'mean':>7}{'min':>7}{'zero':>5}{'cpu':>6}  fails")
    for k in sorted(g):
        rs = g[k]; p = [r["points"] for r in rs]
        print(f"{k[:47]:<48}{len(p):>3}{np.mean(p):>7.1f}{min(p):>7.1f}{sum(x==0 for x in p):>5}"
              f"{np.mean([r.get('cpu',0) for r in rs]):>6.0f}  {sum(r['status']!='PASSED' for r in rs)}")
