"""Jobs for 26 Nov 2026 (run with experiments/run_queue.py): what is left of queue_2026_11_25.py after a fresh restore.

The 25 Nov overnight outputs were never pushed and that machine is gone, so the play-map jobs are rerun here.
Left out on purpose:
- Free Lunch explorers (fractal / fractal_shuffled): the Free Lunch idea is now about how to embed the actions; it
  comes after the life run or after LEXA.
- base_s{42,43,44}: don't rerun baselines; compare against committed results/paper_eval_rgm/seed4{2,3,4}.
- risk_base_s42_screen, risk_play_s42_screen: done and committed in results/risk_screen/{base,play}_s42.

  python experiments/run_queue.py experiments/queue_2026_11_26.py --parallel 4
"""
import runpy
import pathlib

q = runpy.run_path(str(pathlib.Path(__file__).with_name('queue_2026_11_25.py')))
SKIP = {'base_s42', 'base_s43', 'base_s44', 'risk_base_s42_screen', 'risk_play_s42_screen'}
FREE_LUNCH = ('fractal',)  # matches fractal and fractal_shuffled jobs

JOBS = []
for j in q['JOBS']:
    if j['name'] in SKIP or any(f in j['name'] for f in FREE_LUNCH):
        continue
    # dependencies on skipped (already committed) jobs count as done
    j = dict(j, after=[d for d in j.get('after', []) if d not in SKIP])
    JOBS.append(j)
