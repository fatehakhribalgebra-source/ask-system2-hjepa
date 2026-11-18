"""Planned, pooled contrasts across conditions (Welch t-test + Mann-Whitney U, Cohen's d).
Writes results/contrasts.json. Usage: python experiments/contrasts.py
"""
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(env, key):
    g = defaultdict(list)
    for f in (ROOT / 'results' / env).glob('*_seed*.json'):
        d = json.loads(f.read_text())
        for ep in d['episodes']:
            g[d['args']['goal_source']].append(key(ep))
    return g


def pool(g, names):
    return np.concatenate([g[n] for n in names if n in g]) if any(n in g for n in names) else np.array([])


def contrast(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b)
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return dict(mean_a=float(a.mean()), mean_b=float(b.mean()), n_a=len(a), n_b=len(b),
                welch_p=float(t.pvalue), mwu_p=float(u.pvalue), cohens_d=float(d))


EFE = ['pattern:fractal', 'pattern:fractal_shuffled', 'pattern:lorenz', 'pattern:levy', 'pattern:noise_gauss']
NO_EFE_MODEL = ['direct:fractal', 'direct:fractal_shuffled', 'random_macro']
STRUCTURED = ['pattern:fractal', 'pattern:lorenz', 'pattern:levy']
UNSTRUCTURED = ['pattern:fractal_shuffled', 'pattern:noise_gauss']

out = {}
for env, key in [('fourroom', lambda e: 100 * e['coverage']),
                 ('crafter', lambda e: e['tiles']),
                 ('crafter_achievements_ex_wakeup',
                  lambda e: sum(v for k, v in e['achievements'].items() if k != 'wake_up'))]:
    g = load('crafter' if env.startswith('crafter') else env, key)
    out[env] = {
        'EFE selection (pattern:*) vs no selection (direct:*, random_macro)': contrast(pool(g, EFE), pool(g, NO_EFE_MODEL)),
        'structured patterns vs unstructured (shuffled/noise), both with EFE': contrast(pool(g, STRUCTURED), pool(g, UNSTRUCTURED)),
        'System 2 (pattern:*) vs random actions': contrast(pool(g, EFE), pool(g, ['random_actions'])),
        'fractal vs shuffled fractal (EFE)': contrast(pool(g, ['pattern:fractal']), pool(g, ['pattern:fractal_shuffled'])),
    }
    if env == 'fourroom':
        out[env]['v2 fractal vs v2 no pattern'] = contrast(pool(g, ['v2:fractal']), pool(g, ['v2:none']))
        out[env]['v2 fractal learned vs frozen readout'] = contrast(pool(g, ['v2:fractal']), pool(g, ['v2:fractal_nolearn']))
        out[env]['v2 (all) vs v1 System 2 (all)'] = contrast(
            pool(g, ['v2:fractal', 'v2:none', 'v2:fractal_shuffled', 'v2:fractal_nolearn']), pool(g, EFE))
    if env.startswith('crafter'):
        out[env]['Free Lunch literal mapping vs random actions'] = contrast(
            pool(g, ['fl_direct:fractal', 'fl_direct:fractal_shuffled']), pool(g, ['random_actions']))

for env, cs in out.items():
    print(f'\n== {env}')
    for name, c in cs.items():
        if c:
            print(f'  {name}: {c["mean_a"]:.2f} vs {c["mean_b"]:.2f} (n={c["n_a"]}/{c["n_b"]}), '
                  f'Welch p={c["welch_p"]:.3g}, MWU p={c["mwu_p"]:.3g}, d={c["cohens_d"]:+.2f}')
(ROOT / 'results' / 'contrasts.json').write_text(json.dumps(out, indent=1))
