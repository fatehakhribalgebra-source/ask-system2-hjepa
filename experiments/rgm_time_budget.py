"""Where do the env steps go? For each eval JSON: step of the first goal-state handover (RGM phase) and the rest
(H-JEPA's own planner finishing near the goal), for solved and failed runs.

  python experiments/rgm_time_budget.py results/paper_eval_rgm/seed42 results/paper_eval_rgm/seed43 ...
"""
import glob
import json
import os
import sys

import numpy as np

rows = []
for dr in sys.argv[1:]:
    for f in glob.glob(os.path.join(dr, '*.json')):
        d = json.load(open(f))
        if 'decisions' not in d or not d['decisions']:
            continue
        first_goal = next((x['step'] for x in d['decisions'] if x.get('reason') == 'at_goal'), None)
        rows.append((d['success'], d['steps_to_success'], first_goal))
ok = [r for r in rows if r[0]]
bad = [r for r in rows if not r[0]]
print(f'{len(rows)} runs, {len(ok)} solved')
reached = [r for r in ok if r[2] is not None]
print(f'solved: RGM reached the goal state in {len(reached)}/{len(ok)}; '
      f'median step reached {np.median([r[2] for r in reached]):.0f}, median finish {np.median([r[1] for r in ok]):.0f}, '
      f'median finishing phase {np.median([r[1] - r[2] for r in reached]):.0f} steps')
fb = [r for r in bad if r[2] is not None]
print(f'failed: RGM reached the goal state in {len(fb)}/{len(bad)} (then H-JEPA did not finish); '
      f'median step reached {np.median([r[2] for r in fb]) if fb else float("nan"):.0f}')
print('solved finish-step histogram:', np.histogram([r[1] for r in ok], bins=[0, 150, 200, 250, 300, 350])[0].tolist(),
      '(bins 0-150-200-250-300-350)')
