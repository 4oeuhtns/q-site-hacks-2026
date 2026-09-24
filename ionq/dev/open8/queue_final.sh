cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "tag reg3_v3" >/dev/null; do sleep 20; done
S='[{"family":"file","path":"quantum_duel_work/quantum-duel-8q-open-0.7.1/attacks.json#0"}]'
python3 - <<'PY'
import json
a=json.load(open('quantum_duel_work/quantum-duel-8q-open-0.7.1/attacks.json'))
for i,t in enumerate(a): json.dump(t,open(f'dev/open8/specs/final_{i}.json','w'))
json.dump([{"family":"file","path":f"dev/open8/specs/final_{i}.json"} for i in range(2)],open('dev/open8/specs_final.json','w'))
PY
python3 dev/open8/lab.py --defender quantum_duel_work/quantum-duel-8q-open-0.7.1/my_solution/main_open.py --specs @dev/open8/specs_final.json --draws 1 2 3 --tag final_v3 -j 6 > dev/open8/data/final_v3.log 2>&1 &
python3 dev/open8/lab.py --defender stock --specs @dev/open8/specs_final.json --draws 1 2 --tag final_stock -j 2 > dev/open8/data/final_stock.log 2>&1
wait
for w in "0 1" "1 1" "0 2" "1 2"; do echo "$w 1200"; done | xargs -P 4 -L 1 sh -c 'python3 dev/open8/strong_adv.py $0 $1 $2' > dev/open8/data/strong_adv.log 2>&1
echo done
