"""Goal bank: remembered REAL observations (image + proprioception) indexed by their level-3 latent.

H-JEPA reaches goals reliably only when the goal is an observation encoded at every level (the paper's path).
Any latent target an agent picks (a place, an imagined waypoint) is snapped to the nearest remembered real
observation, which is then handed to H-JEPA's unmodified goal-reaching solver. Source: the AntMaze dataset frames
whose z3 we extracted (one per level-3 step); the agent's own observations can be appended with add().
"""
from __future__ import annotations

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch


class GoalBank:
    def __init__(self, h5_path, npz_path, svae, device='cuda'):
        d = np.load(npz_path)
        Z, EP = d['z'], d['ep']
        f = h5py.File(h5_path, 'r')
        off = f['ep_offset'][:]
        rows = np.zeros(len(Z), np.int64)
        for e in np.unique(EP):
            idx = np.flatnonzero(EP == e)
            rows[idx] = off[e] + 20 * np.arange(len(idx))
        self.f, self.rows, self.svae, self.dev = f, rows, svae, device
        self.z = torch.from_numpy(Z).float().to(device)
        with torch.no_grad():
            self.y = svae.zn(self.z)
        self.extra = []                              # (z, pixels, proprio) appended online

    def nearest(self, z_or_y, whitened=False):
        y = z_or_y if whitened else self.svae.zn(z_or_y.reshape(1, -1))[0]
        i = int(((self.y - y.reshape(1, -1)) ** 2).sum(-1).argmin())
        r = int(self.rows[i])
        return dict(z3=self.z[i], pixels=np.asarray(self.f['pixels'][r], np.uint8),
                    proprio=np.asarray(self.f['proprio'][r], np.float32), index=i)
