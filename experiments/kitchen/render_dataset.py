"""Render every state of the D4RL kitchen mixed-v2 episodes (Minari) to JPEG frames, in parallel.

partial-v2 and mixed-v2 hold the same 621 episodes (156,560 steps) with different goal labels, so mixed-v2 is
rendered once; complete-v2 (19 episodes) is small and rendered too. One shard per worker:
data/kitchen/frames<size>/<dataset>_wK.npz with
  ep (N,), t (N,), jpg (N,) object array of JPEG bytes, obs (N, 59), act (N, 9; zeros on the last frame), goal keys.

  python experiments/kitchen/render_dataset.py --workers 6
"""
import argparse
import io
import multiprocessing as mp
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]


def worker(args):
    wid, nw, dataset, size, out = args
    os.environ.setdefault('MINARI_DATASETS_PATH', str(ROOT / 'data' / 'minari'))
    os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')
    import gymnasium as gym
    import gymnasium_robotics
    import minari
    import numpy as np
    from PIL import Image
    gym.register_envs(gymnasium_robotics)
    ds = minari.load_dataset(dataset)
    env = gym.make('FrankaKitchen-v1', render_mode='rgb_array', width=size, height=size)
    env.reset(seed=0)
    u = env.unwrapped
    E, T, J, O, A = [], [], [], [], []
    t0 = time.time()
    for ei, ep in enumerate(ds.iterate_episodes()):
        if ei % nw != wid:
            continue
        obs = np.asarray(ep.observations['observation'], dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        act = np.concatenate([act, np.zeros((1, act.shape[1]), np.float32)], 0)
        for t, o in enumerate(obs):
            u.robot_env.set_state(np.concatenate([o[0:9], o[18:39]]), np.concatenate([o[9:18], o[39:59]]))
            buf = io.BytesIO()
            Image.fromarray(u.render()).save(buf, format='JPEG', quality=95)
            J.append(buf.getvalue()); E.append(ei); T.append(t)
        O.append(obs); A.append(act)
        if wid == 0:
            print(f'w0 ep {ei} frames {len(J)} {time.time() - t0:.0f}s', flush=True)
    env.close()
    jpg = np.empty(len(J), dtype=object); jpg[:] = J
    np.savez(out, ep=np.array(E, np.int32), t=np.array(T, np.int32), jpg=jpg,
             obs=np.concatenate(O), act=np.concatenate(A))
    return len(J)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['D4RL/kitchen/mixed-v2', 'D4RL/kitchen/complete-v2'])
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    od = ROOT / 'data' / 'kitchen' / f'frames{a.size}'
    od.mkdir(parents=True, exist_ok=True)
    for d in a.datasets:
        tag = d.split('/')[-1]
        jobs = [(w, a.workers, d, a.size, str(od / f'{tag}_w{w}.npz')) for w in range(a.workers)
                if not (od / f'{tag}_w{w}.npz').exists()]
        t0 = time.time()
        with mp.get_context('spawn').Pool(len(jobs) or 1) as pool:
            n = sum(pool.map(worker, jobs)) if jobs else 0
        print(f'{d}: {n} frames in {time.time() - t0:.0f}s -> {od}', flush=True)


if __name__ == '__main__':
    main()
