"""E0 (DESIGN_ANIMAL step E0): run-to-run spread on one machine, from deterministic repeats.

Reads results/e0_repro/<planner>/<run>/<planner>_taskNN.json written by
  eval_l3_planner.py --deterministic --repeat R --out results/e0_repro/<planner>/rep<R>
and prints, per planner: solved per repeat, mean +- SE over repeats, tasks whose outcome flips between repeats,
and (if a rep0_again run exists) whether the same seed reproduces the same run step for step.

  python experiments/e0_summary.py [--root results/e0_repro] [--md results/e0_repro/summary.md]
"""
import argparse
import json
import math
import pathlib

import numpy as np


def load(run_dir, planner):
    out = {}
    for f in sorted(run_dir.glob(f'{planner}_task*.json')):
        r = json.loads(f.read_text())
        out[int(r['task'])] = (bool(r['success']), float(r['steps_to_success']))
    return out


def sign_test_p(wins, losses):
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return float('nan')
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def speed_section(all_reps, budget):
    """Paired speed: same task, same repeat (same settled start pose), two planners."""
    out = []
    names = sorted(all_reps)
    for i, p in enumerate(names):
        for q in names[i + 1:]:
            pairs = [(t, r) for r in set(all_reps[p]) & set(all_reps[q])
                     for t in set(all_reps[p][r]) & set(all_reps[q][r])]
            if not pairs:
                continue
            both = [(all_reps[p][r][t][1], all_reps[q][r][t][1]) for t, r in pairs
                    if all_reps[p][r][t][0] and all_reps[q][r][t][0]]
            capped = np.array([[all_reps[x][r][t][1] if all_reps[x][r][t][0] else budget for x in (p, q)]
                               for t, r in pairs])
            out += [f'## Speed: {p} vs {q} (paired by task and repeat, {len(pairs)} pairs)', '']
            if both:
                b = np.array(both)
                d = b[:, 0] - b[:, 1]
                pw, qw = int((d < 0).sum()), int((d > 0).sum())
                out += [f'- both solved: {len(b)} pairs; median steps {p} {np.median(b[:, 0]):.0f} vs {q} '
                        f'{np.median(b[:, 1]):.0f}; mean {b[:, 0].mean():.0f} vs {b[:, 1].mean():.0f}',
                        f'- faster in those pairs: {p} {pw}, {q} {qw}, ties {len(d) - pw - qw} '
                        f'(sign test p = {sign_test_p(pw, qw):.3f})',
                        f'- median step difference {p} - {q}: {np.median(d):+.0f}']
            cd = capped[:, 0] - capped[:, 1]
            se = cd.std(ddof=1) / np.sqrt(len(cd)) if len(cd) > 1 else float('nan')
            out += [f'- time to goal with a failure counted as {budget} steps: {p} {capped[:, 0].mean():.0f}, '
                    f'{q} {capped[:, 1].mean():.0f} (paired difference {cd.mean():+.0f} +- {se:.0f} SE)', '']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/e0_repro')
    ap.add_argument('--md', default='')
    ap.add_argument('--budget', type=int, default=350, help='episode step budget; a failure counts as this many steps')
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    lines = ['# E0: run-to-run spread (deterministic repeats, one machine)', '']
    all_reps = {}
    for pdir in sorted(p for p in root.iterdir() if p.is_dir()):
        planner = pdir.name
        reps = {d.name: load(d, planner) for d in sorted(pdir.iterdir()) if d.is_dir()}
        main_reps = {k: v for k, v in reps.items() if k.startswith('rep') and k[3:].isdigit()}
        tasks = sorted(set.intersection(*(set(v) for v in main_reps.values()))) if main_reps else []
        if not tasks:
            continue
        lines += [f'## {planner} ({len(tasks)} tasks)', '',
                  '| task | ' + ' | '.join(main_reps) + ' | flips |', '|---' * (len(main_reps) + 2) + '|']
        flips = 0
        for t in tasks:
            cells = [f'{int(main_reps[k][t][1])}' if main_reps[k][t][0] else 'x' for k in main_reps]
            outcomes = {main_reps[k][t][0] for k in main_reps}
            flip = len(outcomes) > 1
            flips += flip
            lines.append(f'| {t} | ' + ' | '.join(cells) + f' | {"yes" if flip else ""} |')
        rates = np.array([100 * np.mean([main_reps[k][t][0] for t in tasks]) for k in main_reps])
        se = rates.std(ddof=1) / np.sqrt(len(rates)) if len(rates) > 1 else float('nan')
        solved = ', '.join(f'{k}: {sum(main_reps[k][t][0] for t in tasks)}/{len(tasks)}' for k in main_reps)
        lines += ['', f'- solved per repeat: {solved}',
                  f'- success {rates.mean():.1f} +- {se:.1f} % (mean +- SE over {len(rates)} repeats)',
                  f'- tasks that flip between repeats: {flips}/{len(tasks)}']
        if 'rep0_again' in reps and 'rep0' in reps:
            common = sorted(set(reps['rep0']) & set(reps['rep0_again']))
            same = sum(reps['rep0'][t] == reps['rep0_again'][t] for t in common)
            lines.append(f'- same seed and repeat run twice: {same}/{len(common)} tasks identical (success and steps)')
        lines.append('')
        all_reps[planner] = main_reps
    lines += speed_section(all_reps, a.budget)
    text = '\n'.join(lines)
    print(text)
    if a.md:
        pathlib.Path(a.md).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
