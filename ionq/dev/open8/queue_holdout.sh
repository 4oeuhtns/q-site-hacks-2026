#!/bin/sh
# Held-out set, paired: v11 and v12 validated ZIPs, run concurrently (same machine load).
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
W=/private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/holdout
rm -rf $W; mkdir -p $W/v11 $W/v12
unzip -q quantum_duel_work/quantum-duel-8q-open-0.7.1/submission_v11_validated.zip -d $W/v11
unzip -q quantum_duel_work/quantum-duel-8q-open-0.7.1/submission_v12_validated.zip -d $W/v12
python3 dev/open8/lab.py --defender $W/v11/main.py --specs @dev/open8/specs_holdout.json --draws 1 --tag holdout_v11 -j 4 > dev/open8/data/holdout_v11.log 2>&1 &
python3 dev/open8/lab.py --defender $W/v12/main.py --specs @dev/open8/specs_holdout.json --draws 1 --tag holdout_v12 -j 4 > dev/open8/data/holdout_v12.log 2>&1 &
wait
tail -1 dev/open8/data/holdout_v11.log; tail -1 dev/open8/data/holdout_v12.log
