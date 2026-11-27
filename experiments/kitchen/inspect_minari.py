"""Download the D4RL kitchen datasets (Minari) and print what each episode stores.

  set MINARI_DATASETS_PATH=data/minari
  python experiments/kitchen/inspect_minari.py
"""
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('MINARI_DATASETS_PATH', str(ROOT / 'data' / 'minari'))
import minari  # noqa: E402

IDS = ['D4RL/kitchen/complete-v2', 'D4RL/kitchen/partial-v2', 'D4RL/kitchen/mixed-v2']


def show(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            show(v, f'{pre}{k}.')
    else:
        a = np.asarray(x)
        print(f'    {pre[:-1]}: {a.shape} {a.dtype}')


for i in IDS:
    minari.download_dataset(i)
    ds = minari.load_dataset(i)
    eps = list(ds.iterate_episodes())
    steps = sum(len(e.actions) for e in eps)
    print(f'== {i}: {len(eps)} episodes, {steps} steps, env {ds.spec.env_spec.id if ds.spec.env_spec else None}')
    e = eps[0]
    print('  observations:'); show(e.observations)
    print('  actions:', np.asarray(e.actions).shape)
    print('  infos:'); show(e.infos)
