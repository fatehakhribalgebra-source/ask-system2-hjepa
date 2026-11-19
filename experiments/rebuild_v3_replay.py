"""Rebuild a camera-only v3 replay as camera | whole-maze map (same layout as the other AntMaze replays).

Uses the saved camera video (every 2nd env step) and the last episode's trajectory from the result JSON
(older runs logged xy every 10 steps; positions in between are linearly interpolated). No goal star: the
decoded goal was not stored for those runs.
  python experiments/rebuild_v3_replay.py results/antmaze_v3/fractal_learn_seed0.json
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

import imageio
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
from antmaze_explore import Recorder  # noqa: E402
from system2.hjepa_bridge import HJEPA_DIR  # noqa: E402,F401  (sets MUJOCO_GL, sys.path)
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world import _find_env_method  # noqa: E402


def main(path):
    path = pathlib.Path(path)
    d = json.loads(path.read_text())
    ep = d['episodes'][-1]
    tag = path.stem.rsplit('_seed', 1)[0]
    video = path.with_name(f'replay_{path.stem}.mp4')
    cam = imageio.mimread(video, memtest=False)
    traj = np.asarray(ep['traj'], dtype=np.float64)
    every = max(1, round(d['args']['steps'] / len(traj)))
    world = swm.World(env_name='swm/OGBAntMaze-v0', num_envs=1, max_episode_steps=10, history_size=1, frame_skip=1,
                      ob_type='pixels', width=64, height=64, camera_name='back', maze_type='medium',
                      dataset_type='explore', terminate_at_goal=False, image_shape=(64, 64), verbose=0)
    loco = _find_env_method(world.envs.unwrapped.envs[0], 'neutral_info_from_xy').__self__
    rec = Recorder(loco)
    cells = set()
    for i, f in enumerate(cam):
        t = 2 * i
        x = np.interp(t / every, np.arange(len(traj)), traj[:, 0]); y = np.interp(t / every, np.arange(len(traj)), traj[:, 1])
        cells.add(tuple(np.asarray(loco.env.xy_to_ij(np.array([x, y]))).tolist()))
        rec.frame(np.ascontiguousarray(f[..., :3]).astype(np.uint8), np.array([x, y]), None,
                  f'{tag} ep{ep["episode"]} t={t} cells={len(cells)}/{ep["total_cells"]}')
    world.close()
    rec.save(video)
    print('rebuilt', video, len(rec.frames), 'frames')


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
