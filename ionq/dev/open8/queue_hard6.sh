cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "tag reg5_v5" >/dev/null; do sleep 20; done
python3 dev/open8/lab.py --defender quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py --specs @dev/open8/specs_hard6.json --draws 1 2 --tag hard6_v6 -j 7 > dev/open8/data/hard6_v6.log 2>&1
tail -1 dev/open8/data/hard6_v6.log
