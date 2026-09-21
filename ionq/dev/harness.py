"""Local development harness for Quantum Duel: load attacks, run a defender, score it.

This is a DEVELOPMENT tool only. It needs to know the attack, and your defender
never sees the attack, so nothing here can be imported from `main.py`. Keep it
out of SOURCE_FILES.

Three layers, cheapest first:

  score_patch(patch, attack, rules)   pure algebra, instant, needs no shots
  run_case(case, rules, defender)     a full LocalSession encounter under budget
  scoreboard(cases, rules, defender)  every attack x every seed, as a table

The competition definition of process infidelity for an n-qubit register is

    eps(P, A) = 1 - |Tr(P A)|^2 / 4^n

which is 0 exactly when P inverts A up to a global phase, on every input state.

Usage:
    python3 dev/harness.py                            # baseline defender, all attacks
    python3 dev/harness.py --defender path/to/main.py
    python3 dev/harness.py --seeds 41 73 --attack 4q_conjugated
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Every matrix here is 16x16, so BLAS threading buys nothing and actively fights
# process-level parallelism. Set before numpy loads; spawned workers re-import
# this module, so they inherit it too.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

DEV_DIR = Path(__file__).resolve().parent
ATTACK_DIR = DEV_DIR / "attacks"


def _ensure_sdk():
    """Put the notebook's extracted SDK on sys.path so this runs from anywhere."""
    for base in (DEV_DIR, *DEV_DIR.parents):
        candidate = base / "_quantum_duel_sdk_0_7_2"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("Run the notebook's Setup cell first: _quantum_duel_sdk_0_7_2 was not found.")


_ensure_sdk()

from duelkit.quantum import from_data, infidelity, unitary, validate  # noqa: E402
from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.local import LocalSession  # noqa: E402
from qduel_sdk.rules import Rules  # noqa: E402
from qduel_sdk.templates import instantiate  # noqa: E402


# ---------------------------------------------------------------- scoring

def as_unitary(circuit, qubits):
    """Accept a gate tuple, a circuit_data list of dicts, or an already-built matrix."""
    if isinstance(circuit, np.ndarray):
        return circuit
    circuit = tuple(circuit)
    if circuit and isinstance(circuit[0], dict):
        circuit = from_data(list(circuit))
    return unitary(circuit, qubits)


def process_infidelity(patch, attack, qubits):
    """eps(P, A) -- 0.0 is a perfect inverse, 1.0 is maximally wrong."""
    return infidelity(as_unitary(patch, qubits) @ as_unitary(attack, qubits))


def recovery_points(error, rules=None):
    """Map process infidelity onto the 0-100 scale: flat at both ends, log in between.

    Every factor of 10 you shave off the error is worth the same number of points.
    """
    rules = rules or Rules()
    if not math.isfinite(error) or error < 0:
        raise ValueError("Process infidelity must be a finite number in [0, 1].")
    if error <= rules.good_error:
        return 100.0
    if error >= rules.bad_error:
        return 0.0
    return 100.0 * math.log(rules.bad_error / error) / math.log(rules.bad_error / rules.good_error)


def points_to_infidelity(points, rules=None):
    """Inverse of recovery_points: the error you must reach to earn `points`.

    Useful for planning. On Cup 1, 90 points needs eps <= 0.00158, so an angle
    estimate roughly within 0.08 rad on a single rzz.
    """
    rules = rules or Rules()
    if not 0.0 <= points <= 100.0:
        raise ValueError("Points must lie in [0, 100].")
    return rules.bad_error * (rules.good_error / rules.bad_error) ** (points / 100.0)


def score_patch(patch, attack, rules=None):
    """Grade one patch against one attack: error, points, and whether it is legal.

    An over-budget patch is rejected at submit time, so the runner would keep
    your previous patch instead. `legal` False means this would never be scored.
    """
    rules = rules or Rules()
    error = process_infidelity(patch, attack, rules.qubits)
    try:
        resources = validate(tuple(patch), **rules.validation_kwargs("patch"))
        legal, reason = True, ""
    except (ValueError, TypeError) as exc:
        resources, legal, reason = None, False, str(exc)
    return {
        "process_infidelity": error,
        "recovery_points": recovery_points(error, rules),
        "legal": legal,
        "rejection_reason": reason,
        "resources": resources,
    }


def encounter_recovery(checkpoint_errors, rules=None):
    """Average the per-checkpoint points. A decent early patch is counted three times."""
    checkpoint_errors = list(checkpoint_errors)
    if not checkpoint_errors:
        raise ValueError("An encounter needs at least one checkpoint error.")
    return float(np.mean([recovery_points(e, rules) for e in checkpoint_errors]))


def match_score(my_recovery, their_recovery):
    """Your score in a reciprocal duel: defence and offence weighted equally.

    Defence is how well you undid their attack. Offence is how badly they failed
    to undo yours, which is why a hard-to-learn attack is worth as much as a good
    defender.
    """
    for value in (my_recovery, their_recovery):
        if not 0.0 <= value <= 100.0:
            raise ValueError("Recovery points must lie in [0, 100].")
    return (my_recovery + (100.0 - their_recovery)) / 2.0


# ------------------------------------------------------------ loading attacks

@dataclass(frozen=True)
class AttackCase:
    """One concrete opponent: a fixed gate list, plus where it came from."""
    name: str
    circuit: tuple
    seed: int
    kind: str      # "template" (angles rolled from ranges) or "circuit" (fixed angles)
    source: str

    @property
    def label(self):
        """Filename stem plus the draw seed -- the same string `--attack` filters on."""
        stem = Path(self.source).stem if self.source != "<inline>" else self.name
        return stem if self.kind == "circuit" else f"{stem}@{self.seed}"


def _is_template(payload):
    """Templates are objects with a `gates` list; concrete circuits are bare lists."""
    return isinstance(payload, dict) and "gates" in payload


def parse_attack(payload, *, name, source="<inline>", seeds=(41,), rules=None):
    """Turn one parsed JSON payload into concrete AttackCase objects.

    A template is parameterised, so it yields one case per seed -- exactly how
    the server draws a fresh opponent for every encounter. A concrete circuit has
    fixed angles, so seeds are irrelevant and it yields a single case.
    """
    rules = rules or Rules()
    if _is_template(payload):
        template = Template.model_validate(payload)
        return [AttackCase(payload.get("name", name), instantiate(template, int(seed), rules),
                           int(seed), "template", source)
                for seed in seeds]
    if isinstance(payload, list):
        return [AttackCase(name, tuple(from_data(payload)), 0, "circuit", source)]
    raise ValueError(f"{source}: expected a template object or a list of gates.")


def load_attack(path, *, seeds=(41,), rules=None):
    """Read one .json attack file and return its concrete cases."""
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_attack(payload, name=path.stem, source=path.name, seeds=seeds, rules=rules)


def load_attack_library(directory=ATTACK_DIR, *, seeds=(41,), rules=None, pattern="*.json"):
    """Load every attack file that fits this round's register size.

    Files written for a different qubit count are skipped rather than raising, so
    one directory can hold both four- and eight-qubit opponents.
    """
    rules = rules or Rules()
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No attack directory at {directory}")
    cases, skipped = [], []
    for path in sorted(directory.glob(pattern)):
        try:
            cases.extend(load_attack(path, seeds=seeds, rules=rules))
        except ValueError as exc:
            skipped.append((path.name, str(exc).split("\n")[0][:80]))
    return cases, skipped


# ----------------------------------------------------------- running defenders

def load_defender(main_py):
    """Import run() from a main.py, re-executing it so your latest edits are picked up.

    Modules the file imports (a duel_lib package, say) are purged afterwards, so
    editing a helper between runs takes effect without restarting Python.
    """
    path = Path(main_py).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No defender at {path}")
    before = set(sys.modules)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("harness_defender", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["harness_defender"] = module
        spec.loader.exec_module(module)
        if not callable(getattr(module, "run", None)):
            raise ValueError(f"{path} does not define a callable run(client, rules).")
        return module.run
    finally:
        sys.path.remove(str(path.parent))
        for name in set(sys.modules) - before:
            del sys.modules[name]


def baseline_defender(client, rules):
    """The SDK's public learner, so the harness is useful before you write your own."""
    if rules.qubits == 8:
        from duelkit.recovery8.adapter import run_defender
        return run_defender(client, rules)
    from duelkit.recovery4.adapter import run_defender
    return run_defender(client, rules, grouped=True)


def run_case(case, rules=None, defender=baseline_defender, *, session_seed=None):
    """Play one full encounter: fresh LocalSession, defender runs, scores come back.

    A crash is captured rather than raised -- a harness that dies on the first bad
    attack cannot tell you which attacks are bad.
    """
    rules = rules or Rules()
    seed = case.seed + 100_000 if session_seed is None else session_seed
    session = LocalSession(case.circuit, rules, seed=seed)
    try:
        value = defender(session.client(), rules)
        if inspect.isawaitable(value) or inspect.isgenerator(value):
            raise ValueError("run must execute synchronously.")
        session.client().finish()
        result = session.result()
        errors = [c["process_infidelity"] for c in result["checkpoint_scores"]]
        return {
            "case": case.label, "status": "PASSED", "error": "",
            "recovery_points": result["recovery_points"],
            "checkpoint_points": [c["recovery_points"] for c in result["checkpoint_scores"]],
            "checkpoint_errors": errors,
            "final_infidelity": errors[-1],
            "spent_shots": result["spent_shots"],
            "settings": result["distinct_settings"],
        }
    except BaseException as exc:
        return {
            "case": case.label, "status": "FAILED",
            "error": f"{type(exc).__name__}: {str(exc)[:120]}",
            "recovery_points": 0.0, "checkpoint_points": [], "checkpoint_errors": [],
            "final_infidelity": float("nan"),
            "spent_shots": session.status()["spent_shots"], "settings": 0,
        }


_WORKER_DEFENDER = {}


def _resolve_defender(path):
    """Load (and cache) a defender inside a worker process."""
    if path is None:
        return baseline_defender
    if path not in _WORKER_DEFENDER:
        _WORKER_DEFENDER[path] = load_defender(path)
    return _WORKER_DEFENDER[path]


def _run_one(payload):
    """Top-level so it is picklable by ProcessPoolExecutor."""
    case, rules_dict, defender_path = payload
    return run_case(case, Rules(**rules_dict), _resolve_defender(defender_path))


def scoreboard(cases, rules=None, defender=baseline_defender, *, verbose=True,
               jobs=1, defender_path=None):
    """Run every case and print a table. Returns the rows for further analysis.

    jobs > 1 runs cases in separate processes. Cases are fully independent -- each
    builds its own LocalSession -- so this is embarrassingly parallel. Pass
    defender_path instead of defender when parallelising: a function loaded from a
    file cannot be pickled, so workers load it themselves.
    """
    rules = rules or Rules()
    if jobs and jobs > 1 and len(cases) > 1:
        return _scoreboard_parallel(cases, rules, defender_path, jobs, verbose)
    rows = []
    for index, case in enumerate(cases, 1):
        if verbose:
            print(f"  [{index}/{len(cases)}] {case.label} ...", end="", flush=True)
        row = run_case(case, rules, defender)
        rows.append(row)
        if verbose:
            print(f" {row['status']}  {row['recovery_points']:6.2f} pts")
    if verbose:
        print()
        print(f"{'attack':<22}{'status':<9}{'points':>8}{'infidelity':>13}{'shots':>8}{'settings':>10}")
        print("-" * 70)
        for row in rows:
            print(f"{row['case'][:21]:<22}{row['status']:<9}{row['recovery_points']:>8.2f}"
                  f"{row['final_infidelity']:>13.3e}{row['spent_shots']:>8}{row['settings']:>10}")
        scored = [r["recovery_points"] for r in rows]
        failures = sum(r["status"] == "FAILED" for r in rows)
        print("-" * 70)
        print(f"{'mean':<22}{'':<9}{np.mean(scored) if scored else 0:>8.2f}"
              f"{'':>13}{'':>8}{'':>10}")
        print(f"{'worst':<22}{'':<9}{min(scored) if scored else 0:>8.2f}")
        if failures:
            print(f"\n{failures} case(s) crashed:")
            for row in rows:
                if row["status"] == "FAILED":
                    print(f"  {row['case']}: {row['error']}")
    return rows


# ------------------------------------------------------------------- cli

def _scoreboard_parallel(cases, rules, defender_path, jobs, verbose):
    """Run cases across a process pool, preserving input order in the output."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = min(int(jobs), len(cases), os.cpu_count() or 1)
    rules_dict = rules.model_dump()
    payloads = [(case, rules_dict, defender_path) for case in cases]
    rows = [None] * len(cases)
    if verbose:
        print(f"  running {len(cases)} cases across {jobs} processes ...")
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, payload): i for i, payload in enumerate(payloads)}
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            try:
                rows[i] = future.result()
            except BaseException as exc:
                rows[i] = {"case": cases[i].label, "status": "FAILED",
                           "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                           "recovery_points": 0.0, "checkpoint_points": [],
                           "checkpoint_errors": [], "final_infidelity": float("nan"),
                           "spent_shots": 0, "settings": 0}
            done += 1
            if verbose:
                print(f"  [{done}/{len(cases)}] {rows[i]['case']} "
                      f"{rows[i]['status']}  {rows[i]['recovery_points']:6.2f} pts", flush=True)
    if verbose:
        _print_summary(rows)
    return rows


def _print_summary(rows):
    print()
    print(f"{'attack':<22}{'status':<9}{'points':>8}{'infidelity':>13}{'shots':>8}{'settings':>10}")
    print("-" * 70)
    for row in rows:
        print(f"{row['case'][:21]:<22}{row['status']:<9}{row['recovery_points']:>8.2f}"
              f"{row['final_infidelity']:>13.3e}{row['spent_shots']:>8}{row['settings']:>10}")
    scored = [r["recovery_points"] for r in rows]
    failures = sum(r["status"] == "FAILED" for r in rows)
    print("-" * 70)
    print(f"{'mean':<22}{'':<9}{np.mean(scored) if scored else 0:>8.2f}")
    print(f"{'worst':<22}{'':<9}{min(scored) if scored else 0:>8.2f}")
    if failures:
        print(f"\n{failures} case(s) crashed:")
        for row in rows:
            if row["status"] == "FAILED":
                print(f"  {row['case']}: {row['error']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--defender", default=None,
                        help="path to a main.py; omit to use the SDK baseline learner")
    parser.add_argument("--attacks", default=str(ATTACK_DIR), help="directory of attack .json files")
    parser.add_argument("--attack", default=None, help="run only attacks whose filename contains this")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="run cases in parallel across N processes (0 = one per CPU)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[41],
                        help="template draw seeds; each seed is a separate opponent")
    args = parser.parse_args(argv)

    rules = Rules()
    cases, skipped = load_attack_library(args.attacks, seeds=args.seeds, rules=rules)
    if args.attack:
        cases = [c for c in cases if args.attack in c.source or args.attack in c.name]
    if not cases:
        parser.error(f"No attacks matched in {args.attacks}")

    defender = baseline_defender if args.defender is None else load_defender(args.defender)
    print(f"ruleset : {rules.version} ({rules.qubits} qubits, {rules.total_shots} shots)")
    print(f"defender: {args.defender or 'SDK baseline learner'}")
    print(f"attacks : {len(cases)} case(s) from {args.attacks}, seeds {args.seeds}\n")
    for name, reason in skipped:
        print(f"  skipped {name}: {reason}")

    jobs = (os.cpu_count() or 1) if args.jobs == 0 else args.jobs
    rows = scoreboard(cases, rules, defender, jobs=jobs, defender_path=args.defender)
    return 0 if all(r["status"] == "PASSED" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
