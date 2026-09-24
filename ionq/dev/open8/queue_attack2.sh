cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "tag sweep1_f2def" >/dev/null; do sleep 20; done
python3 dev/open8/lab.py --defender quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py --specs @dev/open8/specs_attack2.json --draws 1 2 --tag attack2_v2 -j 8 > dev/open8/data/attack2_v2.log 2>&1
