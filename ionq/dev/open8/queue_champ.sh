#!/bin/sh
# One chain: wait for the edits run (its queue PID), then the organizers' checks, then the p3 polish test.
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while kill -0 20056 2>/dev/null; do sleep 15; done
L=dev/open8/data/champ_check
mkdir -p $L
for c in local zz mixed multilayer; do
  python3 dev/open8/champ_check.py public $c --seeds 41 > $L/public_$c.log 2>&1 &
done
for s in 41 42 43; do
  python3 dev/open8/champ_check.py public frame --seeds $s > $L/public_frame_$s.log 2>&1 &
done
python3 dev/open8/champ_check.py own --seeds 73 > $L/own_73.log 2>&1 &
wait
echo champ done
cat /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/polish_jobs.txt | xargs -P 8 -L 1 sh -c 'python3 dev/open8/t_polish.py $0 $1 $2 $3 $4' >> dev/open8/data/polish.log 2>&1
echo polish done
