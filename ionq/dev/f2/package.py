"""Build and smoke-test the Practice Cup 2 submission ZIP, as the notebook does.

    python3 dev/f2/package.py                 # build candidate.zip + smoke (frame, merged)
    python3 dev/f2/package.py --final         # also write submission.zip + validation report
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '_quantum_duel_sdk_0_7_2'))
from qduel_sdk.rules import FRAME2_RULESET  # noqa: E402
from qduel_sdk.profiles import profile_rules  # noqa: E402
from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.templates import qualify  # noqa: E402
from qduel_sdk.submission import build_submission, validate_solution, smoke_submission  # noqa: E402
from qduel_sdk.serialization import digest  # noqa: E402

RULES = profile_rules(FRAME2_RULESET)
WORK = ROOT / 'quantum_duel_work' / RULES.version
SOLUTION_DIR = WORK / 'my_solution'
SOURCE_FILES = ['main.py', 'f2def.py']
ATTACK_SOURCES = [ROOT / 'dev/attacks8/ours_2.json', ROOT / 'dev/attacks8/ours_0.json']
NAMES = ['Compile-hard frame A', 'Compile-hard frame B']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    ap.add_argument('--cases', nargs='+', default=['frame', 'merged'])
    ap.add_argument('--version', default='v1')
    args = ap.parse_args()
    attacks = []
    for src, name in zip(ATTACK_SOURCES, NAMES):
        t = json.loads(src.read_text())
        t['name'] = name
        attacks.append(t)
    (WORK / 'attacks.json').write_text(json.dumps(attacks, indent=2), encoding='utf-8')
    print('qualify:', qualify([Template(**a) for a in attacks], RULES)['status'])
    cand = build_submission(SOLUTION_DIR, attacks, WORK / 'candidate.zip', version=args.version,
                            rules=RULES, files=SOURCE_FILES)
    sha = hashlib.sha256(cand.read_bytes()).hexdigest()
    smoke = smoke_submission(cand, rules=RULES, cases=tuple(args.cases), seeds=(41,),
                             trusted_code=True, timeout=900)
    for x in smoke['cases']:
        print(f"  smoke {x['case']}: {x['status']} points={x.get('recovery_points')} {x.get('error', '')}")
    evidence = {'archive_sha256': sha, 'rules_sha256': digest(RULES.model_dump()),
                'cases': args.cases, 'seeds': [41], 'result': smoke}
    (WORK / 'smoke_evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    assert smoke['status'] == 'PASSED', smoke
    if not args.final:
        return
    sub = build_submission(SOLUTION_DIR, attacks, WORK / 'submission.zip', version=args.version,
                           rules=RULES, files=SOURCE_FILES)
    final_sha = hashlib.sha256(sub.read_bytes()).hexdigest()
    validation = validate_solution(sub, rules=RULES, run_smoke=False)
    assert validation['valid_for_upload'], validation.get('errors')
    validation.update(archive_sha256=final_sha, selected_profile=RULES.version,
                      smoke=smoke if final_sha == sha else 'ZIP bytes differ from smoked candidate')
    (WORK / 'local_validation_report.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print('valid_for_upload:', validation['valid_for_upload'], 'sha256', final_sha,
          'identical_to_smoked:', final_sha == sha)
    print('upload:', sub)


if __name__ == '__main__':
    main()
