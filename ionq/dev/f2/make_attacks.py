"""Turn the top skeletons from skel_search into frame2 attack templates.

Writes dev/attacks8/ours_<rank>.json for the top K by verified (else climb) cost,
checks each against the frame2 grammar and sampling readiness, and qualifies
the first two as a submission pair.

    python3 dev/f2/make_attacks.py --top 4
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skel_search  # noqa: E402  (sets up sys.path)
from qduel_sdk.rules import FRAME2_RULESET  # noqa: E402
from qduel_sdk.profiles import profile_rules  # noqa: E402
from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.templates import validate_template, sampling_readiness, qualify  # noqa: E402

RULES = profile_rules(FRAME2_RULESET)
ATTACK8 = skel_search.ROOT / 'dev/attacks8'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skeletons', default=str(skel_search.ROOT / 'dev/results8/skeletons.json'))
    ap.add_argument('--top', type=int, default=4)
    ap.add_argument('--lo', type=float, default=0.75)
    ap.add_argument('--hi', type=float, default=0.9)
    args = ap.parse_args()
    found = json.loads(Path(args.skeletons).read_text())
    found.sort(key=lambda r: tuple(r.get('verified') or r['cost']), reverse=True)
    made = []
    for rank, r in enumerate(found[:args.top]):
        t = skel_search.template(r['skeleton'], f'ours_{rank} (climb seed {r["seed"]})', args.lo, args.hi)
        tm = Template(**t)
        validate_template(tm, RULES)
        ready = sampling_readiness(tm, RULES)
        path = ATTACK8 / f'ours_{rank}.json'
        path.write_text(json.dumps(t, indent=1))
        made.append(t)
        print(path.name, 'cost', r.get('verified') or r['cost'], 'readiness', ready['qualifying'], '/', ready['draws'])
    if len(made) >= 2:
        print(json.dumps(qualify([Template(**made[0]), Template(**made[1])], RULES), indent=1)[:400])


if __name__ == '__main__':
    main()
