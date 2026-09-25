cd /Users/aoeuhtns/Documents/q-site-hacks-2026/ionq
while pgrep -f "tag reg4_v4" >/dev/null; do sleep 20; done
: > dev/open8/data/backload.log
for a in p2 p1; do for g in 36 48 60; do for sd in "21 1" "22 1" "23 1"; do echo "$a $g $sd 96000 400"; done; done; done \
  | xargs -P 6 -L 1 sh -c 'python3 dev/open8/t_frontier.py $0 $1 $2 $3 $4 $5' >> dev/open8/data/backload.log 2>&1
echo done
