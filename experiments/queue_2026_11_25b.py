"""H-JEPA's own 3-level gradient planner on the paper protocol, re-run on this machine (50 tasks x 3 seeds).

Needed because the curious play map hands most decisions back to this planner (no path on a fragmented map): its
score is the floor that any RGM map must beat, measured with the same package versions.
"""
import sys

PY = sys.executable
ALL = [str(t) for t in range(50)]
JOBS = [dict(name=f'gd_s{s}', cmd=[PY, 'experiments/eval_l3_planner.py', '--planner', 'gd', '--seed', str(s),
                                   '--tasks', *ALL, '--out', f'results/paper_eval_play/gd_seed{s}'])
        for s in (42, 43, 44)]
