"""Practice Cup 2 (8q frame2) scoreboard on top of harness.py.

The public frame2 bank is written to dev/attacks8/ once, so our own templates
can sit beside it as plain JSON files.

Usage:
    python3 dev/frame2.py                                   # stock defender, all attacks8
    python3 dev/frame2.py --defender path/to/main.py -j 8 --seeds 41 73
    python3 dev/frame2.py --attack merged
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402  (puts the SDK on sys.path)

from qduel_sdk.rules import FRAME2_RULESET  # noqa: E402
from qduel_sdk.profiles import profile_rules, practice_bank  # noqa: E402

ATTACK8_DIR = harness.DEV_DIR / "attacks8"
RULES = profile_rules(FRAME2_RULESET)


def export_public_bank():
    ATTACK8_DIR.mkdir(exist_ok=True)
    for key, template in practice_bank(RULES).items():
        path = ATTACK8_DIR / f"public_{key}.json"
        if not path.exists():
            path.write_text(json.dumps(template, indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--defender", default=None)
    parser.add_argument("--attacks", default=str(ATTACK8_DIR))
    parser.add_argument("--attack", default=None)
    parser.add_argument("--jobs", "-j", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41])
    parser.add_argument("--out", default=None, help="write rows as JSON here")
    args = parser.parse_args(argv)

    export_public_bank()
    cases, skipped = harness.load_attack_library(args.attacks, seeds=args.seeds, rules=RULES)
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    if args.attack:
        cases = [c for c in cases if args.attack in c.label]
    defender_path = str(Path(args.defender).resolve()) if args.defender else None
    jobs = args.jobs or 8
    defender = harness.baseline_defender if defender_path is None else harness.load_defender(defender_path)
    rows = harness.scoreboard(cases, RULES, defender, jobs=jobs, defender_path=defender_path)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
