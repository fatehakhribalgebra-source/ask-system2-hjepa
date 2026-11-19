"""Whole-maze replay rebuilt from a logged trajectory (for episodes whose camera frames were not saved).

  python experiments/replay_from_traj.py results/antmaze_aif/aif_seed0.json --episode 0
Left panel says the camera was not recorded; right panel = the maze with the path, same layout as other replays.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
from antmaze_explore import Recorder  # noqa: E402
from system2.hjepa_bridge import HJEPA_DIR  # noqa: E402,F401
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world import _find_env_method  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('json'); ap.add_argument('--episode', type=int, default=0)
    a = ap.parse_args()
    path = pathlib.Path(a.json); d = json.loads(path.read_text()); ep = d['episodes'][a.episode]
    traj = np.asarray(ep['traj']); every = max(1, round(d['args']['steps'] / len(traj)))
    world = swm.World(env_name='swm/OGBAntMaze-v0', num_envs=1, max_episode_steps=10, history_size=1, frame_skip=1,
                      ob_type='pixels', width=64, height=64, camera_name='back', maze_type='medium',
                      dataset_type='explore', terminate_at_goal=False, image_shape=(64, 64), verbose=0)
    loco = _find_env_method(world.envs.unwrapped.envs[0], 'neutral_info_from_xy').__self__
    import cv2
    blank = np.full((64, 64, 3), 40, np.uint8)
    rec, cells = Recorder(loco), set()
    for i, xy in enumerate(traj):
        cells.add(tuple(np.asarray(loco.env.xy_to_ij(xy)).tolist()))
        rec.frame(blank, xy, None, f'{path.stem} ep{a.episode} t={i * every} cells={len(cells)}')
        cv2.putText(rec.frames[-1], 'camera not recorded', (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    world.close()
    out = path.with_name(f'replay_{path.stem}_ep{a.episode}_map.mp4')
    rec.save(out); print('saved', out, len(rec.frames), 'frames, cells', len(cells))


if __name__ == '__main__':
    main()
