import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opendef import run_defender  # noqa: E402


def run(client, rules):
    return run_defender(client, rules)
