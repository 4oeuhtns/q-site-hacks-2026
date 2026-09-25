#!/bin/sh
# v12 regression from a frozen snapshot of my_solution (edits cannot leak in mid-run).
# Order: the edit map (target of v12), then the v11 set (non-regression), then fuzz (CPU / false triggers).
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
S=/private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v12b
rm -rf $S; mkdir -p $S
cp quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/*.py quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/tlib.json $S/
L=dev/open8/data
python3 dev/open8/lab.py --defender $S/main_open.py --specs @dev/open8/specs_edits2.json --draws 1 --tag edits2_v12b -j 8 > $L/edits2_v12b.log 2>&1
echo "edits2 $(tail -1 $L/edits2_v12b.log)"
python3 dev/open8/lab.py --defender $S/main_open.py --specs @dev/open8/specs_v11.json --draws 1 2 --tag v12b -j 8 > $L/v12b.log 2>&1
echo "v12b $(tail -1 $L/v12b.log)"
python3 dev/open8/lab.py --defender $S/main_open.py --specs @dev/open8/specs_fuzz.json --draws 1 --tag fuzz_v12b -j 8 > $L/fuzz_v12b.log 2>&1
echo "fuzz $(tail -1 $L/fuzz_v12b.log)"
