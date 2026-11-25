"""Tiny job queue: run named commands with dependencies, at most N at a time (GPU sharing), one log per job.

Jobs are read from a Python file defining JOBS = [dict(name=..., cmd=[...], after=[...]), ...]. A job whose log ends
with the line 'JOB DONE' is skipped, so the queue can be restarted after a crash or a reboot.

  python experiments/run_queue.py experiments/queue_2026_11_25.py --parallel 5
"""
from __future__ import annotations

import argparse
import os
import pathlib
import runpy
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def done(log):
    return log.exists() and log.read_text(encoding='utf-8', errors='replace').rstrip().endswith('JOB DONE')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('--parallel', type=int, default=5)
    ap.add_argument('--only', nargs='*', default=None, help='run only these jobs (and nothing that depends on others)')
    a = ap.parse_args()
    jobs = runpy.run_path(a.jobs)['JOBS']
    if a.only:
        jobs = [j for j in jobs if j['name'] in a.only]
    logs = ROOT / 'logs' / 'queue'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', HJEPA_HOME=str(ROOT / 'data'),
               HOME=os.environ.get('HOME', os.environ.get('USERPROFILE', '')))
    state = {j['name']: ('done' if done(logs / f"{j['name']}.log") else 'wait') for j in jobs}
    running = {}
    while any(s in ('wait', 'run') for s in state.values()):
        for name, (p, fh) in list(running.items()):
            if p.poll() is not None:
                fh.write(f'\nexit code {p.returncode}\n' + ('JOB DONE\n' if p.returncode == 0 else 'JOB FAILED\n'))
                fh.close()
                state[name] = 'done' if p.returncode == 0 else 'failed'
                print(time.strftime('%H:%M:%S'), name, state[name], flush=True)
                del running[name]
        for j in jobs:
            if len(running) >= a.parallel:
                break
            n = j['name']
            if state[n] != 'wait':
                continue
            deps = [state.get(d, 'done') for d in j.get('after', [])]
            if any(d == 'failed' for d in deps):
                state[n] = 'failed'
                print(time.strftime('%H:%M:%S'), n, 'skipped (dependency failed)', flush=True)
                continue
            if all(d == 'done' for d in deps):
                fh = open(logs / f'{n}.log', 'w', encoding='utf-8')
                cmd = [str(c) for c in j['cmd']]
                fh.write(' '.join(cmd) + '\n'); fh.flush()
                running[n] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT), fh)
                state[n] = 'run'
                print(time.strftime('%H:%M:%S'), n, 'started', flush=True)
        time.sleep(20)
    print('queue finished:', {k: v for k, v in state.items()}, flush=True)


if __name__ == '__main__':
    main()
