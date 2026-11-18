"""Convert CrafterPlay (CrafterDojo, CC0) episodes into an H-JEPA HDF5 dataset.

Reads only .npz (allow_pickle=False) and .mp4 files. Writes $HJEPA_HOME/<name>.h5 with flat per-step
columns: pixels (64x64x3 uint8), action (14-d continuous embedding), action_id, achievements (22),
player_position, inventory (12), episode_idx, step_idx, ep_len, ep_offset.
Usage: python experiments/build_crafter_h5.py --per-source 60 --name crafter_play_train
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import cv2
import h5py
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.crafter_actions import DIM, embed  # noqa: E402

RAW = ROOT / 'data' / 'crafter' / 'raw' / 'playdata_raw'
INV = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling', 'wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
       'wood_sword', 'stone_sword', 'iron_sword']


def episodes(source: str):
    base = RAW / source / 'Craftax-Classic-Symbolic-v1'
    for npz in sorted((base / 'state').glob('episode_*.npz')):
        vid = base / 'video' / npz.name.replace('episode_', 'video_').replace('.npz', '.mp4')
        if vid.exists():
            yield npz, vid


def read_video(path: pathlib.Path, size: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    return np.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-source', type=int, default=60)
    ap.add_argument('--skip', type=int, default=0, help='skip the first N episodes per source')
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--name', default='crafter_play_train')
    ap.add_argument('--max-len', type=int, default=10000)
    a = ap.parse_args()
    out = ROOT / 'data' / f'{a.name}.h5'
    eps = []
    for src in ('survival-10B', 'gatherer-10B'):
        eps += list(episodes(src))[a.skip:a.skip + a.per_source]
    print(f'{len(eps)} episodes -> {out}')
    with h5py.File(out, 'w') as f:
        S = a.size
        cols = {
            'pixels': f.create_dataset('pixels', (0, S, S, 3), np.uint8, maxshape=(None, S, S, 3),
                                       chunks=(64, S, S, 3), compression='lzf'),
            'action': f.create_dataset('action', (0, DIM), np.float32, maxshape=(None, DIM), chunks=(4096, DIM)),
            'action_id': f.create_dataset('action_id', (0,), np.int64, maxshape=(None,), chunks=(4096,)),
            'achievements': f.create_dataset('achievements', (0, 22), np.bool_, maxshape=(None, 22), chunks=(4096, 22)),
            'player_position': f.create_dataset('player_position', (0, 2), np.float32, maxshape=(None, 2), chunks=(4096, 2)),
            'inventory': f.create_dataset('inventory', (0, 12), np.float32, maxshape=(None, 12), chunks=(4096, 12)),
            'episode_idx': f.create_dataset('episode_idx', (0,), np.int64, maxshape=(None,), chunks=(4096,)),
            'step_idx': f.create_dataset('step_idx', (0,), np.int64, maxshape=(None,), chunks=(4096,)),
        }
        lens, offs, n = [], [], 0
        for ei, (npz, vid) in enumerate(eps):
            z = np.load(npz, allow_pickle=False)
            px = read_video(vid, S)
            T = min(len(px), len(z['action']) + 1, a.max_len)
            act = np.zeros(T, dtype=np.int64)
            act[:T - 1] = z['action'][:T - 1]                     # action taken after obs t; last padded
            data = dict(pixels=px[:T], action=embed(act), action_id=act,
                        achievements=z['achievements'][:T], player_position=z['player_position'][:T].astype(np.float32),
                        inventory=np.stack([z[f'inventory_{k}'][:T] for k in INV], -1).astype(np.float32),
                        episode_idx=np.full(T, ei), step_idx=np.arange(T))
            for k, ds in cols.items():
                ds.resize(n + T, axis=0)
                ds[n:n + T] = data[k]
            lens.append(T); offs.append(n); n += T
            if ei % 10 == 0:
                print(f'  {ei}/{len(eps)} episodes, {n} frames', flush=True)
        f.create_dataset('ep_len', data=np.array(lens))
        f.create_dataset('ep_offset', data=np.array(offs))
    print(f'done: {len(lens)} episodes, {n} frames')


if __name__ == '__main__':
    main()
