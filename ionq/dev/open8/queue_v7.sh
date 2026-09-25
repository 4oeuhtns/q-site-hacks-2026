cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "tag probe7_v6" >/dev/null || pgrep -f queue_probe7 >/dev/null; do sleep 20; done
python3 dev/open8/lab.py --defender quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py --specs @dev/open8/specs_v7.json --draws 1 2 --tag v7 -j 8 > dev/open8/data/v7.log 2>&1
tail -1 dev/open8/data/v7.log
