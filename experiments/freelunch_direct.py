"""Free Lunch Exp. V-A: simple mappings of pattern labels to N/E/S/W, no learning, no world model.

Protocol (Ertle, Levin & Scheutz 2025, Sec. IV-C, V-A, Table I):
- 60 patterns per family, 20x20 labels (400 steps), read in raster order, 4 labels.
- Fractals: Halley on x(x^a - b), a in [2,6], b in [1,12] -> exactly 60 fractals.
- Controls: a pixel shuffle of each pattern (same label histogram) and uniform noise.
- 4 mappings per pattern; the best mapping (by % explored) is kept per pattern.
- Maze: Kruskal 11x11, path scale 2; pattern i runs on maze seed i (seeds [0, 60)).
- Metric: % of path squares visited; stats: one-way ANOVA + Tukey HSD.
Usage: python experiments/freelunch_direct.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.freelunch_maze import FreeLunchMaze  # noqa: E402
from system2 import patterns as P  # noqa: E402

MAPPINGS = [[0, 1, 2, 3], [0, 2, 1, 3], [3, 2, 1, 0], [3, 1, 2, 0]]  # label -> action index (N,E,S,W)
RES = 20


def family(name: str, i: int) -> np.ndarray:
    """i-th 20x20 4-label pattern of a family (i in [0, 60))."""
    if name == 'fractal':
        a, b = 2 + i // 12, 1 + i % 12
        return P.halley_fractal(a=a, b=b, res=RES, bounds=((-2, 2), (-2, 2)))
    if name == 'fractal_wide':      # paper's example plot bounds
        a, b = 2 + i // 12, 1 + i % 12
        return P.halley_fractal(a=a, b=b, res=RES, bounds=((-10, 10), (-10, 10)))
    if name == 'fractal_basins':    # coloured by converged root
        a, b = 2 + i // 12, 1 + i % 12
        return P.halley_basins(a=a, b=b, res=RES, bounds=((-2, 2), (-2, 2)))
    if name == 'rule30':
        full = P.rule30(width=64, steps=64)
        lab = full * 2 + np.roll(full, 1, axis=1)
        r, c = (i * 7) % 44, (i * 13) % 44
        return lab[r:r + RES, c:c + RES]
    if name == 'life':
        return P.game_of_life(size=2 * RES, steps=RES, seed=i)[:, :RES]
    if name == 'noise':
        return np.random.default_rng(1000 + i).integers(0, 4, size=(RES, RES))
    raise ValueError(name)


def run(labels: np.ndarray, maze_seed: int, mapping: list[int]) -> tuple[float, float]:
    flat = labels.reshape(-1)
    env = FreeLunchMaze(seed=maze_seed, max_steps=len(flat))
    env.reset()
    moved = 0
    for lab in flat:
        _, _, _, info = env.step(mapping[int(lab) % 4])
        moved += info['moved']
    return env.coverage(), moved / len(flat)


def best_mapping(labels, maze_seed):
    res = [run(labels, maze_seed, m) for m in MAPPINGS]
    return max(res, key=lambda r: r[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=60)
    ap.add_argument('--families', nargs='+', default=['fractal', 'rule30', 'life'])
    ap.add_argument('--out', default=str(ROOT / 'results' / 'freelunch_direct.json'))
    a = ap.parse_args()

    groups: dict[str, list[float]] = {}
    for fam in a.families:
        for shuf in (False, True):
            key = fam + ('_shuffled' if shuf else '')
            covs = []
            for i in range(a.n):
                lab = family(fam, i)
                if shuf:
                    lab = P.shuffled(lab, seed=i)
                covs.append(best_mapping(lab, maze_seed=i)[0])
            groups[key] = covs
    groups['noise'] = [best_mapping(family('noise', i), i)[0] for i in range(a.n)]
    rng = np.random.default_rng(0)
    groups['random_walk'] = [run(rng.integers(0, 4, size=(RES, RES)), i, MAPPINGS[0])[0] for i in range(a.n)]

    print(f'{"group":20s} {"% explored":>12s}')
    for k, v in groups.items():
        print(f'{k:20s} {100 * np.mean(v):8.1f} ± {100 * np.std(v) / np.sqrt(len(v)):.1f}')
    names = list(groups)
    F, p = stats.f_oneway(*groups.values())
    print(f'\none-way ANOVA: F={F:.2f}, p={p:.2e}')
    tk = stats.tukey_hsd(*groups.values())
    pairs = {}
    for fam in a.families:
        for other in (fam + '_shuffled', 'noise'):
            i, j = names.index(fam), names.index(other)
            pairs[f'{fam} vs {other}'] = float(tk.pvalue[i, j])
            print(f'Tukey HSD {fam:8s} vs {other:18s} p={tk.pvalue[i, j]:.3g}  '
                  f'diff={100 * (np.mean(groups[fam]) - np.mean(groups[other])):+.1f} pts')
    out = dict(groups={k: dict(mean=float(np.mean(v)), se=float(np.std(v) / np.sqrt(len(v))), values=v)
                       for k, v in groups.items()}, anova=dict(F=float(F), p=float(p)), tukey=pairs)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
