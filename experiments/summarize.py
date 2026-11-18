"""Aggregate exploration results (Free Lunch style): per-condition mean ± SE, one-way ANOVA, Tukey HSD,
and the key planned comparisons (pattern vs its shuffle, pattern vs noise, EFE vs direct).
Writes results/summary_<env>.md and a bar chart PNG.
Usage: python experiments/summarize.py antmaze|crafter
"""
from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(env: str):
    groups = defaultdict(list)
    extra = defaultdict(list)
    for f in sorted((ROOT / 'results' / env).glob('*_seed*.json')):
        d = json.loads(f.read_text())
        cond = d['args']['goal_source']
        for ep in d['episodes']:
            if env != 'crafter':
                groups[cond].append(100 * ep['coverage'])
                extra[cond].append(ep.get('rooms', 0))
            else:
                groups[cond].append(ep['n_achievements'])
                extra[cond].append(ep['tiles'])
                extra[cond + '|steps'].append(ep['steps'])
            extra[cond + '|motion'].append(100 * ep['motion'])
    return groups, extra


def main(env: str):
    groups, extra = load(env)
    if not groups:
        print('no results yet'); return
    metric = {'antmaze': '% maze cells explored', 'fourroom': '% arena cells explored'}.get(
        env, 'achievements unlocked / episode')
    names = sorted(groups, key=lambda k: -np.mean(groups[k]))
    third = ' tiles visited | steps survived |' if env == 'crafter' else ' rooms reached (of 4) |'
    lines = [f'# {env}: {metric}', '', '| condition | n | mean | SE | % time in motion |' + third,
             '|---|---|---|---|---|---|' + ('---|' if env == 'crafter' else '')]
    for n in names:
        v = np.asarray(groups[n]); m = np.asarray(extra[n + '|motion'])
        row = f'| {n} | {len(v)} | {v.mean():.2f} | {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f} | {m.mean():.0f} |'
        if env == 'crafter':
            row += f' {np.mean(extra[n]):.0f} | {np.mean(extra[n + "|steps"]):.0f} |'
        else:
            row += f' {np.mean(extra[n]):.1f} |'
        lines.append(row)
    valid = [n for n in names if len(groups[n]) > 1]
    if len(valid) > 2:
        F, p = stats.f_oneway(*[groups[n] for n in valid])
        lines += ['', f'One-way ANOVA across {len(valid)} conditions: F = {F:.2f}, p = {p:.3g}', '']
        tk = stats.tukey_hsd(*[groups[n] for n in valid])
        pairs = [('pattern:fractal', 'pattern:fractal_shuffled'), ('pattern:fractal', 'pattern:noise_gauss'),
                 ('pattern:fractal', 'direct:fractal'), ('direct:fractal', 'direct:fractal_shuffled'),
                 ('pattern:fractal', 'random_macro'), ('pattern:fractal', 'random_actions'),
                 ('pattern:lorenz', 'pattern:noise_gauss'), ('pattern:noise_gauss', 'random_macro'),
                 ('random_macro', 'random_actions'), ('fl_direct:fractal', 'fl_direct:fractal_shuffled'),
                 ('v2:fractal', 'v2:none'), ('v2:fractal', 'v2:fractal_shuffled'),
                 ('v2:fractal', 'v2:fractal_nolearn'), ('v2:fractal', 'pattern:fractal'),
                 ('v2:none', 'random_macro'), ('pattern:fractal', 'pattern:levy')]
        lines += ['| comparison | diff | Tukey HSD p |', '|---|---|---|']
        for x, y in pairs:
            if x in valid and y in valid:
                i, j = valid.index(x), valid.index(y)
                lines.append(f'| {x} vs {y} | {np.mean(groups[x]) - np.mean(groups[y]):+.2f} | {tk.pvalue[i, j]:.3g} |')
    if env == 'crafter':
        lines += ['', '## Crafter score (geometric mean of achievement success rates)', '',
                  '| condition | score % | top achievements |', '|---|---|---|']
        for f in sorted((ROOT / 'results' / 'crafter').glob('*_seed*.json')):
            d = json.loads(f.read_text()); s = d['summary']
            top = sorted(s['success_rates'].items(), key=lambda kv: -kv[1])[:4]
            lines.append(f'| {d["args"]["goal_source"]} (seed {d["args"]["seed"]}) | {s["score"]:.2f} | ' +
                         ', '.join(f'{k} {v:.0f}%' for k, v in top if v > 0) + ' |')
    text = '\n'.join(lines)
    (ROOT / 'results' / f'summary_{env}.md').write_text(text)
    print(text)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4))
        means = [np.mean(groups[n]) for n in names]
        ses = [np.std(groups[n], ddof=1) / np.sqrt(len(groups[n])) if len(groups[n]) > 1 else 0 for n in names]
        colors = ['#2a6fdb' if n.startswith('pattern') else '#7aa6f0' if n.startswith('direct') else '#9a9a9a'
                  for n in names]
        ax.bar(range(len(names)), means, yerr=ses, color=colors, capsize=3)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
        ax.set_ylabel(metric); ax.set_title(f'{env}: reward-free exploration (mean ± SE)')
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout(); fig.savefig(ROOT / 'results' / f'summary_{env}.png', dpi=130)
    except Exception as e:  # plotting is optional
        print('plot failed:', e)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'antmaze')
