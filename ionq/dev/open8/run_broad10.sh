cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
D=quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py
python3 dev/open8/lab.py --defender $D --specs @dev/open8/specs_broad10.json --draws 3 --tag broad_v10 -j 8 > dev/open8/data/broad_v10.log 2>&1
python3 dev/open8/lab.py --defender $D --specs @dev/open8/specs_final.json --draws 1 2 3 --tag final_v10 -j 6 > dev/open8/data/final_v10.log 2>&1
tail -1 dev/open8/data/broad_v10.log; tail -1 dev/open8/data/final_v10.log
