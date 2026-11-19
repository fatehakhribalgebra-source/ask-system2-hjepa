"""Run the H-JEPA planning eval with every task in its own Python process.

Why: on Windows MuJoCo renders through glfw (no EGL). After the first World is closed, every World created
later in the same process renders pure black frames, so tasks 2..N planned from black images (old 1/11 result).
planning_eval's chunked eval is resumable (finished chunks are skipped), and HJEPA_ONE_CHUNK_PER_PROCESS
makes a process exit after one new chunk; this driver relaunches until every chunk exists, and the last run
writes metrics.yaml exactly as upstream does.

  python experiments/run_ant_eval_isolated.py --config ant_l3 --model ant_hjepa_l3 --seed 42
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HJ = ROOT / 'H-JEPA' / 'h_jepa'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='ant_l3')
    ap.add_argument('--env', default='ant')
    ap.add_argument('--model', default='ant_hjepa_l3')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='eval_l3')
    ap.add_argument('--extra', nargs='*', default=[], help='extra hydra overrides')
    a = ap.parse_args()
    home = os.environ.get('HJEPA_HOME', str(ROOT / 'data'))
    ckpt = pathlib.Path(home) / 'ckpts' / a.env / a.model / f'seed{a.seed}' / f'{a.model}_object.ckpt'
    res = ckpt.parent / a.out_dir
    env = dict(os.environ, HJEPA_HOME=home, HJEPA_ONE_CHUNK_PER_PROCESS='1', PYTHONUTF8='1',
               HOME=os.environ.get('HOME', os.environ.get('USERPROFILE', '')))
    cmd = [sys.executable, 'eval.py', '--config-name', a.config, f'seed={a.seed}', f'output.dir={a.out_dir}',
           f'policy={ckpt}', *a.extra]
    for i in range(1000):
        if (res / 'metrics.yaml').exists():
            break
        r = subprocess.run(cmd, cwd=HJ, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        lines = [l for l in r.stdout.splitlines() if l.startswith('tasks ') or l.startswith('{')]
        print(*lines, sep='\n', flush=True)
        if r.returncode != 0:
            print(r.stderr[-3000:], flush=True)
            sys.exit(r.returncode)
    print((res / 'metrics.yaml').read_text())


if __name__ == '__main__':
    main()
