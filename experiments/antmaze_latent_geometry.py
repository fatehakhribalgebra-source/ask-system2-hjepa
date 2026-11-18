"""Diagnostic: geometry of H-JEPA latents across AntMaze cells (calibrates System-2 step size)."""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
from system2.hjepa_bridge import build_policy  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from experiments.antmaze_explore import CKPT, loco_of  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def main():
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=42), 'ant_l3')
    cfg.world.max_episode_steps = 50; cfg.world.num_envs = 1
    world = swm.World(**OmegaConf.to_container(cfg.world, resolve=True), image_shape=(64, 64), verbose=0)
    world.set_policy(policy); world.reset(seed=0)
    loco = loco_of(world)
    base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in world.infos.items()}
    Z, IJ = {l: [] for l in range(1, 4)}, []
    for ij in loco.all_cells:
        xy = np.asarray(loco.env.ij_to_xy(ij))
        info = loco.settled_info_from_xy(xy, render_pixels=True)
        d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        d['pixels'] = info['pixels'][None, None]
        d['proprio'] = info['proprio'][None, None]; d['qpos'] = info['qpos'][None, None]; d['qvel'] = info['qvel'][None, None]
        prep = policy._prepare_info(d)
        with torch.no_grad():
            e = solver._encode_all_levels(prep, key='pixels')
        for l in range(1, 4):
            Z[l].append(e[f'embed_{l}'].reshape(-1).cpu().numpy())
        IJ.append(ij)
    IJ = np.array(IJ)
    grid = np.abs(IJ[:, None] - IJ[None]).sum(-1)
    for l in range(1, 4):
        z = np.stack(Z[l]); dist = np.linalg.norm(z[:, None] - z[None], axis=-1)
        print(f'level {l}: D={z.shape[1]} |z| mean {np.linalg.norm(z, axis=1).mean():.2f}  '
              f'per-dim std {z.std(0).mean():.3f}')
        for g in (1, 2, 3, 4, 6):
            m = grid == g
            if m.any():
                print(f'   cells {g} apart (manhattan): latent dist {dist[m].mean():.2f} ± {dist[m].std():.2f}')
        iu = np.triu_indices(len(z), 1)
        print(f'   corr(latent dist, manhattan) = {np.corrcoef(dist[iu], grid[iu])[0, 1]:.3f}')
    np.savez(ROOT / 'results' / 'antmaze_latent_cells.npz', ij=IJ, **{f'z{l}': np.stack(Z[l]) for l in Z})
    world.close()


if __name__ == '__main__':
    main()
