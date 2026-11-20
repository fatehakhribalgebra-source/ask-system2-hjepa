"""Replay videos for the level-3 planner comparison: camera view | maze map with the ant's trail and the goal.

Runs the paper eval through experiments/eval_l3_planner.py (unchanged) with World.step wrapped to record frames.
One process per (planner, task), as the Windows renderer requires. A recorded run is a fresh run, so its outcome can
differ from the table run of the same task.

  python experiments/record_l3_planner.py --tasks 1 6 10 --planners gd rgm_closed rgm_commit
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

# name -> eval_l3_planner arguments
PLANNERS = {
    'gd': ['--planner', 'gd'],
    'tschantz_L15': ['--planner', 'tschantz', '--plan-len', '15'],
    'rgm_closed': ['--planner', 'rgm', '--rgm-mode', 'closed'],
    'rgm_commit': ['--planner', 'rgm', '--rgm-mode', 'commit'],
    'rgm_adaptive': ['--planner', 'rgm', '--rgm-mode', 'adaptive'],
    'rgm_feedback': ['--planner', 'rgm', '--rgm-mode', 'feedback'],
    'rgm_feedback_learn': ['--planner', 'rgm', '--rgm-mode', 'feedback', '--rgm-learn'],
    'rgm_paper_z3': ['--planner', 'rgm_paper', '--rgm-obs-level', '3',
                     '--rgm-paper-path', str(ROOT / 'data' / 'rgm_paper_z3_k64_seed42.pkl')],
    'rgm_every_action': ['--planner', 'rgm', '--rgm-mode', 'feedback', '--rgm-every-action'],
    'rgm_every_action_learn': ['--planner', 'rgm', '--rgm-mode', 'feedback', '--rgm-every-action', '--rgm-learn'],
}


def record_one(name, task, out):
    os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
    os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'
    sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT))
    import numpy as np
    import eval_l3_planner as ev
    from system2.hjepa_bridge import HJEPA_DIR  # noqa: F401  (puts H-JEPA on sys.path)
    import stable_worldmodel as swm
    from stable_worldmodel.world import _find_env_method
    from antmaze_explore import Recorder

    state = dict(rec=None, t=0)
    orig_step = swm.World.step

    def step(self, *args, **kw):
        r = orig_step(self, *args, **kw)
        if state['rec'] is None:
            loco = _find_env_method(self.envs.unwrapped.envs[0], 'neutral_info_from_xy').__self__
            state['rec'] = Recorder(loco)
        info = self.infos
        px = np.asarray(info['pixels']).reshape(-1, 64, 64, 3)[-1].astype(np.uint8)
        xy = np.asarray(info['xy']).reshape(-1)[-2:]
        gxy = np.asarray(info['goal_xy']).reshape(-1)[:2]
        state['t'] += 1
        state['rec'].frame(px, xy, gxy, f'{name}  task {task}  t={state["t"]}  dist={np.linalg.norm(xy - gxy):.1f}')
        return r

    swm.World.step = step
    sys.argv = ['eval_l3_planner.py', '--single', *PLANNERS[name], '--tasks', str(task), '--out', str(out),
                '--tag', '_rec']
    ev.main()
    res = sorted(out.glob(f'*task{task:02d}.json'), key=lambda p: p.stat().st_mtime)[-1]
    r = json.loads(res.read_text())
    tag = 'solved' if r['success'] else 'failed'
    mp4 = out / f'replay_{name}_task{task:02d}_{tag}.mp4'
    state['rec'].save(mp4)
    res.rename(out / f'{name}_task{task:02d}.json')
    print(json.dumps(dict(planner=name, task=task, success=r['success'], steps=r['steps_to_success'],
                          video=str(mp4.relative_to(ROOT)))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, nargs='+', default=[1, 6, 10])
    ap.add_argument('--planners', nargs='+', default=list(PLANNERS))
    ap.add_argument('--out', default=str(ROOT / 'results' / 'l3_planner_pc2' / 'replays'))
    ap.add_argument('--single', nargs=2, metavar=('PLANNER', 'TASK'), help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.single:
        record_one(a.single[0], int(a.single[1]), out)
        return
    env = dict(os.environ, PYTHONUTF8='1', HJEPA_HOME=os.environ.get('HJEPA_HOME', str(ROOT / 'data')),
               HOME=os.environ.get('HOME', os.environ.get('USERPROFILE', '')))
    for name in a.planners:
        for t in a.tasks:
            if list(out.glob(f'replay_{name}_task{t:02d}_*.mp4')):
                continue
            r = subprocess.run([sys.executable, __file__, '--single', name, str(t), '--out', str(out)], env=env,
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
            line = [l for l in r.stdout.splitlines() if l.startswith('{"planner"')]
            print(line[-1] if line else r.stderr[-1500:], flush=True)


if __name__ == '__main__':
    main()
