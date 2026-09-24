# Open final (8q) — running lab notes

Raw data: `data/results.jsonl` (one row per encounter; `summarize.py` aggregates by tag).
Offline learner logs: `data/pursuit_*.log`.

## Defenders
- `f2def` — Cup 2 frame learner (100/100 in the official Cup 2 round). Runs on OPEN8 unchanged.
- `opendef` v1 — f2def + public-architecture template fitter (`tlib.json`, 533 entries), BIC selection.
- `opendef` v2 — v1 + greedy gate pursuit (unknown architecture), 600 s encounter CPU cap.

## Findings so far (2026-09-24)
| attack | f2def | template fitter | pursuit (offline, no template) |
|---|---|---|---|
| notebook default `mixed` (open_example/31, 18g/6e) | 0 | 100 | 100 |
| notebook default `multilayer`/37 (72g/24e) | 0 | 97.9 | 0 (found 54 wrong gates) |
| open_example/23 (36g/12e) | — | 100 | — |
| notebook `open_demo` (5g) | — | — | 100 |
| random sparse 4/1, 8/2, 12/4, 18/6 | — | — | 100 |
| generic 24g/8e continuous | 0 (72g) | — | 100 |
| exact frames k<=2 | 100 | — | — |
| frames k>=6, near-Clifford width>=0.05 | 0 | — | — |

- Pursuit's defeat depth is between 24 and 72 gates (attack2 round maps 36/48).
- f2def regression: Cup 2 frame A scored 100/100/0 (lost checkpoint 3). Uninvestigated.
- f2def CPU up to ~250 s on FRAME4 public example.

## sweep1_f2def complete (118 encounters, 0 crashes)
f2def decodes: exact frames k<=4, pair_ins<=2, public frames, merged, sparse <=4 gates.
f2def zero: k>=6, near-Clifford width>=0.12, all continuous (generic/open_example), sparse >=8 gates.
TIMEOUT RISK: frame k=4 mean 514 CPU s; sparse 4/1 287 s. f2def's own synthesis is not under ENCOUNTER_CAP.
