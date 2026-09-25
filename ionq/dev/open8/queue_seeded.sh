#!/bin/sh
cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "queue_champ.sh" >/dev/null; do sleep 15; done
cat /private/tmp/claude-501/-Users-aoeuhtns-Documents-q-site-hacks-2026/6ff150cb-2f1f-4f19-a202-d964cf9d79aa/scratchpad/seeded_jobs.txt | xargs -P 8 -L 1 sh -c 'python3 dev/open8/t_seeded.py $0 $1 $2 $3' >> dev/open8/data/seeded.log 2>&1
echo seeded done
