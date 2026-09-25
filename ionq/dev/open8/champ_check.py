"""Headless run of the organizers' Championship Submission Check notebook.

Executes the notebook's own helper cells (package check, SDK smoke runner,
own-attack runner, evidence review) without Jupyter, printing tables as text.

    python3 dev/open8/champ_check.py static
    python3 dev/open8/champ_check.py public frame --seeds 41
    python3 dev/open8/champ_check.py own --seeds 73
    python3 dev/open8/champ_check.py review data/champ_checks/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IONQ = HERE.parents[1]
NB = IONQ / "Quantum_Duel_Championship_Prep.ipynb"
ZIP = IONQ / "quantum_duel_work/quantum-duel-8q-open-0.7.1/submission.zip"
OUT = HERE / "data/champ_checks"

# The notebook's embedded SDK is byte-identical to this copy (checked with diff -r).
sys.path.insert(0, str(IONQ / "_quantum_duel_sdk_0_7_2"))


def show_rows(rows, columns):
    if not rows:
        print("No rows recorded.")
        return
    cells = [[str(r.get(k, "")) for k, _ in columns] for r in rows]
    heads = [label for _, label in columns]
    w = [min(60, max(len(h), *(len(c[i]) for c in cells))) for i, h in enumerate(heads)]
    print("  ".join(h[:w[i]].ljust(w[i]) for i, h in enumerate(heads)))
    for c in cells:
        print("  ".join(v[:w[i]].ljust(w[i]) for i, v in enumerate(c)))


def load_notebook():
    cells = json.loads(NB.read_text())["cells"]
    import hashlib, os
    ns = {"__name__": "champ_check", "sys": sys, "os": os, "json": json,
          "hashlib": hashlib, "Path": Path}  # what the notebook's setup cell imports
    # 7: imports + RULES; 13: review helpers; 20: own-attack runner.
    for i in (7, 13, 20):
        exec(compile("".join(cells[i]["source"]), f"notebook-cell-{i}", "exec"), ns)
    ns["show_rows"] = show_rows
    return ns


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["static", "public", "own", "review"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    ns = load_notebook()
    rules = ns["RULES"]
    pre = ns["inspect_candidate"](Path(a.zip), rules)
    print("Static package check:", pre["status"], "| sha256:", pre.get("sha256"))
    if a.mode == "static":
        if pre["valid"]:
            ch = pre["checked"]
            show_rows(ch["archive"]["files"], [("name", "Packaged file"), ("bytes", "Bytes")])
            show_rows([{"slot": k, "name": t["name"], "gates": len(t["gates"]),
                        "ents": sum(len(g["targets"]) == 2 for g in t["gates"])}
                       for k, t in enumerate(ch["attacks"], 1)],
                      [("slot", "Slot"), ("name", "Template"), ("gates", "Gates"), ("ents", "Entanglers")])
        else:
            print(pre["error"])
    elif a.mode == "public":
        ev, path = ns["run_existing_public_tests"](
            Path(a.zip), rules, cases=tuple(a.args), seeds=tuple(a.seeds or (41,)),
            timeout=a.timeout, trusted_code=True, output_dir=OUT, maximum_runs=64)
        print("Evidence:", path)
        ns["print_assessment"](ev, Path(a.zip), rules, pre)
    elif a.mode == "own":
        ev, path = ns["run_existing_own_tests"](
            Path(a.zip), rules, seeds=tuple(a.seeds or (73,)), timeout=a.timeout,
            trusted_code=True, output_dir=OUT, maximum_runs=64)
        print("Evidence:", path)
        ns["print_assessment"](ev, Path(a.zip), rules, pre)
    else:
        for p in a.args:
            print("\n=====", p)
            ns["print_assessment"](ns["load_saved_evidence"](p), Path(a.zip), rules, pre)
