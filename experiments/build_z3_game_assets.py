"""Assets for the z3 game: K real macro-action prototypes and an evaluation-only probe z3 -> xy.

Prototypes: k-means on the 18,750 TRUE level-3 macro-actions, centres mapped through level 3's action-encoder
LayerNorm so every prototype is a valid move F3 was trained on. Probe: ridge regression from z3 to the ant's true xy
(read from the dataset at the frame each z3 encodes); used ONLY to measure how much of the maze an agent's
imagined play covers - the agent never sees it.

  python experiments/build_z3_game_assets.py --k 12
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'z3_game_assets.pt'))
    a = ap.parse_args()
    d = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')
    Z, A, EP = d['z'], d['acts'][:, 0], d['ep']
    # xy of the frame each pair's z3 encodes: pair j of episode e is level-3 step t=j -> env step offset + 20 t
    f = h5py.File(ROOT / 'data' / 'datasets' / 'visual_antmaze_medium_stitch_val.h5', 'r')
    off, xy_all = f['ep_offset'][:], f['xy'][:]
    XY = np.zeros((len(Z), 2), np.float32)
    for e in np.unique(EP):
        idx = np.flatnonzero(EP == e)
        XY[idx] = xy_all[off[e] + 20 * np.arange(len(idx))]
    # k-means on real macro-actions
    rng = np.random.default_rng(a.seed)
    C = A[rng.choice(len(A), a.k, replace=False)].copy()
    for _ in range(50):
        lab = ((A[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([A[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(a.k)])
    model = torch.load(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed42' / 'ant_hjepa_l3_object.ckpt',
                       map_location='cpu', weights_only=False)
    ln = list(model.jepas[-1].action_encoder.children())[-1]
    with torch.no_grad():
        P = ln(torch.from_numpy(C).float())
    counts = np.bincount(lab, minlength=a.k)
    print('prototype norms', P.norm(dim=1).numpy().round(3), '| cluster sizes', counts)
    # ridge probe z3 -> xy, episode split
    eps = np.unique(EP); val = set(rng.choice(eps, len(eps) // 10, replace=False)); va = np.isin(EP, list(val))
    Zm, Ym = Z[~va].mean(0), XY[~va].mean(0)
    Zc = Z[~va] - Zm
    W = np.linalg.solve(Zc.T @ Zc + 10 * np.eye(Z.shape[1]), Zc.T @ (XY[~va] - Ym))
    pred = (Z[va] - Zm) @ W + Ym
    err = np.linalg.norm(pred - XY[va], axis=1)
    print(f'probe z3 -> xy on held-out episodes: mean error {err.mean():.2f} (maze cell = 4 units), '
          f'R^2 {1 - ((pred - XY[va]) ** 2).sum() / ((XY[va] - XY[va].mean(0)) ** 2).sum():.3f}')
    torch.save({'protos': P, 'counts': torch.from_numpy(counts), 'probe_W': torch.from_numpy(W).float(),
                'probe_zm': torch.from_numpy(Zm).float(), 'probe_ym': torch.from_numpy(Ym).float(),
                'z_start': torch.from_numpy(Z).float(), 'xy_start': torch.from_numpy(XY)}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
