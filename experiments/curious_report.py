"""Summary of run_curious.py: per arm and budget, map size and quality, maze coverage of the map, goal-task success.

Coverage uses the eval-only ridge probe z3 -> xy (data/z3_game_assets.pt, mean error ~ a fraction of a cell): the
cells (4 x 4 maze units) holding a REAL state's exemplar, over the cells the dataset ever visits.

  python experiments/curious_report.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def cells(xy, size=4.0):
    return {tuple(c) for c in np.floor(np.asarray(xy) / size).astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    import torch
    A = torch.load(ROOT / 'data' / 'z3_game_assets.pt', map_location='cpu', weights_only=False)
    W, zm, ym = A['probe_W'].numpy(), A['probe_zm'].numpy(), A['probe_ym'].numpy()
    maze = cells(A['xy_start'].numpy())
    base = ROOT / 'results' / 'curious' / f'seed{a.seed}'
    rows = []
    for arm in sorted(p for p in base.iterdir() if p.is_dir()):
        budgets = sorted({int(p.stem[5:]) for p in arm.glob('map_B*.pkl')} |
                         {int(p.name[6:]) for p in arm.glob('eval_B*') if p.is_dir()})
        for b in budgets:
            row = dict(arm=arm.name, budget=b)
            snap = arm / f'map_B{b}.pkl'
            if snap.exists():                                   # map after exploring (budget 0 = empty, none)
                with open(snap, 'rb') as fh:
                    m = pickle.load(fh)
                real = [m.ex[s] for s in range(m.K) if m.real[s]]
                cov = len(cells((np.stack(real) - zm) @ W + ym) & maze) / len(maze) if real else 0.0
                row |= m.map_summary() | dict(coverage=round(cov, 3))
            ev = arm / f'eval_B{b}'
            res = [json.loads(p.read_text()) for p in sorted(ev.glob('*_task*.json'))] if ev.exists() else []
            steps = [r['steps_to_success'] for r in res if r['success']]
            row |= dict(solved=f'{sum(r["success"] for r in res)}/{len(res)}',
                        mean_steps=round(float(np.mean(steps)), 1) if steps else None)
            rows.append(row)
    keys = ['arm', 'budget', 'states', 'real_states', 'edges', 'tried', 'confirmed', 'refuted', 'coverage', 'solved',
            'mean_steps']
    print('| ' + ' | '.join(keys) + ' |\n|' + '---|' * len(keys))
    for r in rows:
        print('| ' + ' | '.join(str(r.get(k, '')) for k in keys) + ' |')


if __name__ == '__main__':
    main()
