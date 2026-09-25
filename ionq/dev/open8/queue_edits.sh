cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "t_frontier.py" >/dev/null; do sleep 15; done
python3 dev/open8/lab.py --defender quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py --specs @dev/open8/specs_edits.json --draws 1 2 --tag edits_v10 -j 8 > dev/open8/data/edits_v10.log 2>&1
tail -1 dev/open8/data/edits_v10.log
