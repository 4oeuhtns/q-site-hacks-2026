"""Does the packaged defender use more than one core? (run from ionq/)

    env -u OMP_NUM_THREADS ... python3 dev/open8/thread_probe.py ZIP_DIR CASE SEED

Runs one LocalSession encounter with the defender from ZIP_DIR and reports wall
time, process CPU time (what ENCOUNTER_CAP measures), their ratio, and the peak
native thread count of this process (sampled with ps every 0.5 s).
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

zip_dir, case, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
IONQ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(IONQ / "_quantum_duel_sdk_0_7_2"))
sys.path.insert(0, zip_dir)

import numpy as np  # noqa: E402
from qduel_sdk.local import LocalSession  # noqa: E402
from qduel_sdk.profiles import profile_rules, practice_bank  # noqa: E402
from qduel_sdk.rules import OPEN8_RULESET  # noqa: E402
from qduel_sdk.contracts import Template  # noqa: E402
from qduel_sdk.templates import instantiate  # noqa: E402

peak = [0]
stop = False


def sample():
    while not stop:
        out = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True).stdout
        peak[0] = max(peak[0], len(out.strip().splitlines()) - 1)
        time.sleep(0.5)


rules = profile_rules(OPEN8_RULESET)
attack = instantiate(Template.model_validate(practice_bank(rules)[case]), seed, rules)
import main  # noqa: E402  (the packaged entry point)

threading.Thread(target=sample, daemon=True).start()
sess = LocalSession(attack, rules, seed=seed + 100000)
w0, c0 = time.time(), time.process_time()
main.run(sess.client(), rules)
sess.client().finish()
wall, cpu = time.time() - w0, time.process_time() - c0
stop = True
res = sess.result()
blas = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                       "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS")}
print(json.dumps(dict(case=case, seed=seed, env=blas, wall=round(wall, 1), cpu=round(cpu, 1),
                      cpu_over_wall=round(cpu / wall, 2), peak_native_threads=peak[0],
                      points=round(res["recovery_points"], 2))), flush=True)
