#!/bin/sh
# Overnight queue, Friday 2026-09-25 (deadline 23:59). Most decision-relevant stage first.
# Encounters run the frozen v11 ZIP, not my_solution/, so morning edits cannot leak in.
# Morning read-out: python3 dev/open8/overnight_report.py
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
L=dev/open8/data
F=$(mktemp -d)/v11
unzip -q quantum_duel_work/quantum-duel-8q-open-0.7.1/submission_v11_validated.zip -d "$F"
cat > "$F/main_cap.py" <<'EOF'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import opendef
opendef.ENCOUNTER_CAP = float(os.environ.get("QD_CAP", opendef.ENCOUNTER_CAP))
def run(client, rules):
    return opendef.run_defender(client, rules)
EOF
echo "start $(date) defender=$F"

# 1. Crash / over-cap hunt: 116 random + edge-case legal attacks.
python3 dev/open8/lab.py --defender "$F/main.py" --specs @dev/open8/specs_fuzz.json --draws 1 --tag fuzz_v11 -j 8 > $L/fuzz_v11.log 2>&1
echo "fuzz done $(date): $(tail -1 $L/fuzz_v11.log)"

# 2. Edited public templates: 5 bases x 15 edit types x 2 variants.
python3 dev/open8/lab.py --defender "$F/main.py" --specs @dev/open8/specs_edits2.json --draws 1 --tag edits2_v11 -j 8 > $L/edits2_v11.log 2>&1
echo "edits2 done $(date): $(tail -1 $L/edits2_v11.log)"

# 3. Timeout insurance: the v11 regression set with ENCOUNTER_CAP 400 s (paired with tag v11).
QD_CAP=400 python3 dev/open8/lab.py --defender "$F/main_cap.py" --specs @dev/open8/specs_v11.json --draws 1 2 --tag v11cap400 -j 8 > $L/v11cap400.log 2>&1
echo "cap400 done $(date): $(tail -1 $L/v11cap400.log)"

# 4. Offline pursuit engines p1/p2/p3 on new generic seeds, 96k shots, 400 s (uses my_solution/pursuit.py).
for a in p1 p2 p3; do for g in 36 48; do for s in 24 25 26 27 28 29; do echo "$a $g $s 1 96000 400"; done; done; done \
  | xargs -P 8 -L 1 sh -c 'python3 dev/open8/t_frontier.py $0 $1 $2 $3 $4 $5' >> $L/frontier_overnight.log 2>&1
echo "frontier done $(date)"
echo "ALL DONE $(date)"
