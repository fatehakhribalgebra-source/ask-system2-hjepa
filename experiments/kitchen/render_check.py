"""Render D4RL kitchen dataset states to pixels and check speed and fidelity.

Observation layout (FrankaKitchen-v1, 59-d): robot qpos 9 | robot qvel 9 | object qpos 21 | object qvel 20.
qpos = obs[0:9] + obs[18:39] (30), qvel = obs[9:18] + obs[39:59] (29). The stored obs carry small observation noise
(robot_noise_ratio 0.01), so a rendered frame is the recorded state up to that noise.

  python experiments/kitchen/render_check.py --out results/kitchen/render_check.png
"""
import argparse
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault('MINARI_DATASETS_PATH', str(ROOT / 'data' / 'minari'))
os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')

import gymnasium as gym  # noqa: E402
import gymnasium_robotics  # noqa: E402,F401
import minari  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

gym.register_envs(gymnasium_robotics)


def make_env(size):
    return gym.make('FrankaKitchen-v1', render_mode='rgb_array', width=size, height=size)


def render_obs(env, obs):
    u = env.unwrapped
    qpos = np.concatenate([obs[0:9], obs[18:39]])
    qvel = np.concatenate([obs[9:18], obs[39:59]])
    u.robot_env.set_state(qpos, qvel)
    return u.render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='D4RL/kitchen/mixed-v2')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'kitchen' / 'render_check.png'))
    a = ap.parse_args()
    ds = minari.load_dataset(a.dataset)
    ep = next(iter(ds.iterate_episodes()))
    obs = np.asarray(ep.observations['observation'])
    env = make_env(a.size)
    env.reset(seed=0)
    idx = np.linspace(0, len(obs) - 1, 8).astype(int)
    t0 = time.time()
    frames = [render_obs(env, obs[i]) for i in idx]
    dt = (time.time() - t0) / len(idx)
    # replay check: from the first state, step the recorded actions and compare object qpos with the dataset
    env.reset(seed=0)
    render_obs(env, obs[0])
    acts = np.asarray(ep.actions)
    err = []
    for k in range(min(50, len(acts))):
        o, *_ = env.step(acts[k])
        err.append(np.abs(o['observation'][:9] - obs[k + 1][:9]).max())
    env.close()
    grid = np.concatenate([np.concatenate(frames[:4], 1), np.concatenate(frames[4:], 1)], 0)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(a.out)
    print(f'frames {frames[0].shape}, {dt * 1000:.1f} ms/frame -> {156_560 * dt / 60:.1f} min for 156,560 frames')
    print(f'replay of recorded actions: max robot-qpos error over 50 steps {max(err):.4f} (first {err[0]:.4f})')
    print('saved', a.out)


if __name__ == '__main__':
    main()
