"""Paired comparison of eval arms on the same (seed, task) pairs, with the agent's mechanism counts on discordant tasks.

  python experiments/compare_runs.py results/paper_eval_pc3 base agent agentp
"""
import collections
import glob
import json
import os
import sys

root, ref, *arms = sys.argv[1:]
R = {}
for v in [ref, *arms]:
    for s in (42, 43, 44):
        for f in glob.glob(os.path.join(root, v, f'seed{s}', 'rgm_paper_task*.json')):
            d = json.load(open(f)); R[(v, s, d['task'])] = d
for v in arms:
    pairs = sorted((s, t) for (vv, s, t) in R if vv == v and (ref, s, t) in R)
    both = sum(R[(v, s, t)]['success'] and R[(ref, s, t)]['success'] for s, t in pairs)
    only_v = [(s, t) for s, t in pairs if R[(v, s, t)]['success'] and not R[(ref, s, t)]['success']]
    only_r = [(s, t) for s, t in pairs if R[(ref, s, t)]['success'] and not R[(v, s, t)]['success']]
    sv = [R[(v, s, t)]['steps_to_success'] for s, t in pairs if R[(v, s, t)]['success'] and R[(ref, s, t)]['success']]
    sr = [R[(ref, s, t)]['steps_to_success'] for s, t in pairs if R[(v, s, t)]['success'] and R[(ref, s, t)]['success']]
    print(f'{v} vs {ref}: {len(pairs)} pairs | both solved {both} | only {v} {len(only_v)} | only {ref} {len(only_r)} | '
          f'mean steps when both solve {sum(sv) / max(len(sv), 1):.0f} vs {sum(sr) / max(len(sr), 1):.0f}')
    for tag, lst in ((f'lost by {v}', only_r), (f'won by {v}', only_v)):
        agg = collections.Counter()
        for s, t in lst:
            agg.update(R[(v, s, t)].get('rgm_stats') or {})
        keys = ('refused', 'efe_fallback', 'f3_choice', 'surprise', 'gate_kept', 'gate_rejected', 'moves_failed')
        print(f'   {tag}: {lst} | summed {dict((k, agg[k]) for k in keys)}')
