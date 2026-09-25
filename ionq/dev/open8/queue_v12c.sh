#!/bin/sh
# Rerun only the encounters hit by the v12 f2def-guard bug (avail shadowing), with the fix.
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
rm -rf /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12c; mkdir -p /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12c
cp quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/*.py quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/tlib.json /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12c/
python3 dev/open8/lab.py --defender /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12c/main_open.py --specs @dev/open8/specs_fix_d1.json --draws 1 --tag fix_v12c -j 8 > dev/open8/data/fix_v12c_d1.log 2>&1
python3 dev/open8/lab.py --defender /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12c/main_open.py --specs @dev/open8/specs_fix_d2.json --draws 2 --tag fix_v12c -j 8 > dev/open8/data/fix_v12c_d2.log 2>&1
tail -1 dev/open8/data/fix_v12c_d1.log; tail -1 dev/open8/data/fix_v12c_d2.log
