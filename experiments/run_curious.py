"""Explore, then reach goals given on the fly (curious RGM, AntMaze).

Phase 1 (no goal): starting from an EMPTY map, the agent explores for E episodes (each = one eval task's start state,
350 env steps, the task's goal image ignored); the map is carried from episode to episode. After each budget in
--budgets the map is snapshotted. Phase 2: from every snapshot, the paper's goal tasks are run with the same planner
(map still carried and learning, the continual setting). Budget 0 = the empty map (pure live play).

  python experiments/run_curious.py --arm curious_dream --explorer curious --dream --budgets 0 10 25 50 --eval-tasks 0 1 ... 12
  python experiments/run_curious.py --arm random_dream  --explorer random  --dream --budgets 0 10 25 50 --eval-tasks ...

Output: results/curious/seed<S>/<arm>/{explore,map_B<b>.pkl,eval_B<b>/}. Summarise with curious_report.py.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / 'experiments' / 'eval_l3_planner.py'


def run(args, tasks, out, persist, explore, pattern_seed=None):
    cmd = [sys.executable, str(EVAL), '--planner', 'rgm_paper', '--rgm-obs-level', '3', '--rgm-curious',
           '--seed', str(args.seed), '--rgm-persist', str(persist), '--rgm-explorer', args.explorer,
           '--rgm-eta', str(args.eta), '--rgm-rho', str(args.rho), '--out', str(out),
           '--tasks', *map(str, tasks)]
    if explore and pattern_seed is not None:
        cmd += ['--rgm-pattern-seed', str(pattern_seed)]
    if args.dream:
        cmd.append('--rgm-dream')
    if args.stream:
        cmd.append('--rgm-stream')
        if explore and args.dump_stream:
            cmd.append('--rgm-dump-stream')
    if explore:
        cmd.append('--rgm-explore')
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--explorer', default='curious',
                    choices=['curious', 'random', 'fractal', 'fractal_shuffled', 'noise_labels'])
    ap.add_argument('--dream', action='store_true')
    ap.add_argument('--stream', action='store_true', help='count a sliding-window transition at every env step')
    ap.add_argument('--dump-stream', action='store_true', help='save each explore episode\'s per-step z3 (stream npz)')
    ap.add_argument('--budgets', type=int, nargs='+', default=[0, 10, 25, 50], help='explore episodes per snapshot')
    ap.add_argument('--explore-tasks', type=int, nargs='+', default=list(range(50)),
                    help='start states of the explore episodes, in order (task goals are ignored)')
    ap.add_argument('--eval-tasks', type=int, nargs='+', default=list(range(50)))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--skip-eval', action='store_true')
    a = ap.parse_args()

    base = ROOT / 'results' / 'curious' / f'seed{a.seed}' / a.arm
    base.mkdir(parents=True, exist_ok=True)
    live = base / 'map_live.pkl'
    done = 0
    for b in sorted(a.budgets):
        snap = base / f'map_B{b}.pkl'
        if b > done:
            if not snap.exists():
                tasks = [a.explore_tasks[i % len(a.explore_tasks)] for i in range(done, b)]
                # explore episodes are numbered so a repeated start state still gets its own result file
                for i, t in zip(range(done, b), tasks):
                    run(a, [t], base / 'explore' / f'ep{i:03d}', live, explore=True,
                        pattern_seed=a.seed * 1000 + i)            # a new pattern per episode (Free Lunch explorer)
                shutil.copy(live, snap)
            elif not live.exists():
                shutil.copy(snap, live)
            done = b
        if a.skip_eval:
            continue
        ev = base / f'eval_B{b}'
        work = ev / 'map.pkl'
        if not work.exists():
            ev.mkdir(parents=True, exist_ok=True)
            if snap.exists():
                shutil.copy(snap, work)
        run(a, a.eval_tasks, ev, work, explore=False)


if __name__ == '__main__':
    main()
