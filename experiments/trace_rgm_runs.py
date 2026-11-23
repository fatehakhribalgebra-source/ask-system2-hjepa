"""Print the RGM's decision trace (state > target, * = new induction, H = handover) for eval JSONs.

  python experiments/trace_rgm_runs.py results/rgm_screen/stuck/seed42 [--tasks 8 27]
"""
import argparse
import collections
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument('dirs', nargs='+')
ap.add_argument('--tasks', type=int, nargs='*')
ap.add_argument('--n', type=int, default=20)
a = ap.parse_args()
for dr in a.dirs:
    for f in sorted(glob.glob(os.path.join(dr, '*.json'))):
        d = json.load(open(f))
        if a.tasks and d['task'] not in a.tasks:
            continue
        c, seq = collections.Counter(), []
        for x in d['decisions'] or []:
            k = 'handover' if x.get('handover') else 'held' if x.get('held') else 'induced' if x.get('induced') else 'kept'
            c[k] += 1
            if x.get('held'):
                continue
            if x.get('handover'):
                seq.append(f"H:{x.get('reason', '')[:4]}@{x.get('s', '?')}")
            else:
                seq.append(f"{x['s']}>{x['target']}" + ('*' if x.get('induced') else '') + ('' if x.get('ctx', True) else '~'))
        res = int(d['steps_to_success']) if d['success'] else 'x'
        print(f"{dr} t{d['task']:02d} {res} {dict(c)}\n   " + ' '.join(seq[:a.n]))
