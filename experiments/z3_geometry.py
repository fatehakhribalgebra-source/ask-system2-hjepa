"""E0b: how good is a level-3 latent space as a MAP? (ASK Animal design, docs/DESIGN_ANIMAL.md section 2b)

Evaluation only: true xy is read from the dataset to score the latents; no agent ever sees it.
Input: a macro-action file from experiments/build_macro_action_data.py (z = the level-3 latent at level-3 step j of
episode ep, the frame at env step ep_offset[ep] + 20 j).

Metrics (Content-Geometry, Ren et al. 2026):
  probe_r2     ridge z3 -> xy on held-out episodes (content: is the position readable?)
  jag_z, jag_xy jaggedness J = path length / path length after a 3-step moving average, along each episode
               (1 = smooth). jag_z / jag_xy > 1 means the latent path is more jagged than the real path.
  rho_maze     Spearman correlation between latent distance (whitened) and maze distance (shortest path over maze
               cells, 4x4 units, cell adjacency taken from the cells the data actually crosses between) over random pairs
  rho_local    the same over pairs at most 3 level-3 steps apart (local geometry, what a reflex hop sees)
  place_spread K=64 k-means places in whitened z3: median over places of the xy spread (std, maze units) of its frames
  place_cells  median number of maze cells a place's frames fall in (1 = a place is one cell)

  python experiments/z3_geometry.py --macro data/macro_actions_ant_l3_h1_seed42.npz --out results/geometry/release_s42.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_paper import kmeans  # noqa: E402

CELL = 4.0


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def jag(P, ep):
    num = den = 0.0
    for e in np.unique(ep):
        p = P[ep == e]
        if len(p) < 5:
            continue
        sm = np.stack([p[i - 1:i + 2].mean(0) for i in range(1, len(p) - 1)])
        num += np.linalg.norm(np.diff(p[1:-1], axis=0), axis=1).sum()
        den += np.linalg.norm(np.diff(sm, axis=0), axis=1).sum()
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--macro', required=True)
    ap.add_argument('--data', default=str(ROOT / 'data' / 'datasets' / 'visual_antmaze_medium_stitch_val.h5'))
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--pairs', type=int, default=20000)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = np.load(a.macro)
    Z, EP = d['z'].astype(np.float32), d['ep']
    f = h5py.File(a.data, 'r')
    off, xy_all = f['ep_offset'][:], f['xy'][:]
    XY = np.zeros((len(Z), 2), np.float32)
    J = np.zeros(len(Z), np.int64)
    for e in np.unique(EP):
        idx = np.flatnonzero(EP == e)
        J[idx] = np.arange(len(idx))
        XY[idx] = xy_all[off[e] + a.stride * np.arange(len(idx))]

    # whitening as in PaperRGM (32-d PCA)
    mean = Z.mean(0)
    ev, V = np.linalg.eigh(np.cov((Z - mean).T))
    proj = (V[:, ::-1][:, :32] / np.sqrt(ev[::-1][:32])).T
    Y = (Z - mean) @ proj.T

    # probe z3 -> xy (held-out episodes)
    eps = np.unique(EP)
    va = np.isin(EP, rng.choice(eps, len(eps) // 10, replace=False))
    Zm, Ym = Z[~va].mean(0), XY[~va].mean(0)
    Zc = Z[~va] - Zm
    W = np.linalg.solve(Zc.T @ Zc + 10 * np.eye(Z.shape[1]), Zc.T @ (XY[~va] - Ym))
    pred = (Z[va] - Zm) @ W + Ym
    r2 = 1 - ((pred - XY[va]) ** 2).sum() / ((XY[va] - XY[va].mean(0)) ** 2).sum()

    # maze graph from the data: cells, and the cell changes the trajectories actually make
    cell = np.floor(xy_all / CELL).astype(np.int64)
    cid = {c: i for i, c in enumerate(sorted(set(map(tuple, cell))))}
    adj = {i: set() for i in cid.values()}
    ci = np.array([cid[tuple(c)] for c in cell])
    same_ep = np.ones(len(ci) - 1, bool); same_ep[off[1:] - 1] = False
    step = np.abs(cell[1:] - cell[:-1]).sum(1)              # grid (Manhattan) distance of consecutive frames
    keep = same_ep & (step == 1)                            # crossings into a 4-neighbour cell
    for u, v in set(zip(ci[:-1][keep].tolist(), ci[1:][keep].tolist())):
        adj[u].add(v); adj[v].add(u)
    n_c = len(cid)
    D = np.full((n_c, n_c), np.inf)
    for s in range(n_c):
        D[s, s] = 0; q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if D[s, v] == np.inf:
                    D[s, v] = D[s, u] + 1; q.append(v)
    zc = np.array([cid.get(tuple(c), -1) for c in np.floor(XY / CELL).astype(np.int64)])

    # global: random pairs
    i, j = rng.integers(len(Z), size=(2, a.pairs))
    ok = (zc[i] >= 0) & (zc[j] >= 0) & np.isfinite(D[zc[i], zc[j]])
    rho_maze = spearman(np.linalg.norm(Y[i[ok]] - Y[j[ok]], axis=1), D[zc[i[ok]], zc[j[ok]]]
                        + np.linalg.norm(XY[i[ok]] - XY[j[ok]], axis=1) / 100)       # tiny xy tie-break
    # local: pairs within 3 level-3 steps of the same episode
    li, lj = [], []
    for k in range(1, 4):
        m = np.flatnonzero((EP[k:] == EP[:-k]))
        li.append(m); lj.append(m + k)
    li, lj = np.concatenate(li), np.concatenate(lj)
    rho_local = spearman(np.linalg.norm(Y[li] - Y[lj], axis=1), np.linalg.norm(XY[li] - XY[lj], axis=1))

    # places
    C = kmeans(Y.astype(np.float32), a.k, seed=a.seed)
    lab = np.concatenate([((Y[s:s + 4096, None] - C[None]) ** 2).sum(-1).argmin(1) for s in range(0, len(Y), 4096)])
    spread = [float(XY[lab == p].std(0).mean()) for p in range(a.k) if (lab == p).sum() > 1]
    ncells = [len(set(zc[lab == p])) for p in range(a.k) if (lab == p).sum() > 1]
    purity = [np.bincount(zc[lab == p][zc[lab == p] >= 0]).max() / (lab == p).sum() for p in range(a.k)
              if (lab == p).sum() > 1]
    # how many distinct places share one maze cell (pose / heading splits a place)
    per_cell = [len(set(lab[zc == c])) for c in range(n_c) if (zc == c).sum() > 20]

    res = dict(macro=a.macro, n=len(Z), episodes=len(eps), probe_r2=round(float(r2), 4),
               jag_z=round(float(jag(Y, EP)), 3), jag_xy=round(float(jag(XY, EP)), 3), rho_maze=round(rho_maze, 4),
               rho_local=round(rho_local, 4), place_spread=round(float(np.median(spread)), 3),
               place_cells=float(np.median(ncells)), place_purity=round(float(np.median(purity)), 3),
               places_per_cell=float(np.median(per_cell)), maze_cells=n_c)
    res['jag_ratio'] = round(res['jag_z'] / res['jag_xy'], 3)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
