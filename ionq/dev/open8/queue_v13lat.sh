#!/bin/sh
# v13 under simulated oracle latency (35 ms per client call), 4 workers.
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
rm -rf /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v13; mkdir -p /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v13
cp quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/*.py quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/tlib.json /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v13/
export LAB_LATENCY_MS=35
python3 dev/open8/lab.py --defender /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v13/main_open.py --specs @dev/open8/specs_lat_extra.json --draws 1 --tag v13lat -j 4 > dev/open8/data/v13lat_extra.log 2>&1
python3 dev/open8/lab.py --defender /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/snap_v13/main_open.py --specs @dev/open8/specs_lat_d12.json --draws 1 2 --tag v13lat -j 4 > dev/open8/data/v13lat.log 2>&1
tail -1 dev/open8/data/v13lat_extra.log; tail -1 dev/open8/data/v13lat.log
