"""System 2, generator G: pattern -> code u -> macro-action VAE decoder -> imagine (H-JEPA L3) -> EFE -> commit.

No reservoir. Every candidate k reads its own pattern stream on every env step (as in v3); at each decision its
last `window` symbols are mapped to the d_u-dim code by a FIXED linear projection, standardised with running
statistics so the codes match the VAE prior N(0, I) in scale. The decoder then produces macro-actions on the
distribution H-JEPA was trained with. Selection, commitment and the ensemble are inherited from v3.
Code sources: a pattern stream (structured or its shuffled control), or stream=None -> u ~ N(0, I) (plain prior).
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .action_vae import ActionVAE
from .goal_engine_v3 import GoalEngineV3, V3Config
from .patterns import PatternStream


class GoalEngineGen(GoalEngineV3):
    def __init__(self, d: int, stream: PatternStream | None, vae: ActionVAE, cfg: V3Config = V3Config(),
                 seed: int = 0, window: int = 16):
        cfg.learn_readout = False                   # the generative model is the learned part
        super().__init__(d, stream, cfg, seed=seed, reservoir_n=8)
        self.vae, self.window = vae, window
        self.bufs = [deque(maxlen=window) for _ in range(cfg.k)]
        g = torch.Generator().manual_seed(seed + 11)
        in_dim = (stream.dim // stream.window) if stream is not None else 1
        self.Pu = (torch.randn(vae.d_u, window * in_dim, generator=g) / np.sqrt(window * in_dim)).to(cfg.device)
        self.u_mean = torch.zeros(vae.d_u, device=cfg.device); self.u_var = torch.ones(vae.d_u, device=cfg.device)
        self.n_codes = 0

    def reset(self, episode_offset: int = 0):
        super().reset(episode_offset)
        for b in self.bufs:
            b.clear()

    def drive(self, z=None):
        """Read one symbol from every candidate's stream (once per env step)."""
        if self.stream is None:
            return
        for k, o in enumerate(self.pos):
            v = self.stream.peek(int(o) - self.stream.pos)
            self.bufs[k].append(v.reshape(self.stream.window, -1)[-1])     # newest symbol of the window
        self.pos += 1

    def _codes(self) -> torch.Tensor:
        dev = self.cfg.device
        if self.stream is None or len(self.bufs[0]) < self.window:
            return torch.randn(self.cfg.k, self.vae.d_u, device=dev)
        x = torch.from_numpy(np.stack([np.concatenate(list(b)) for b in self.bufs])).float().to(dev)
        u = x @ self.Pu.T
        # running standardisation (pattern statistics differ per family; the prior is N(0, I))
        self.n_codes += len(u)
        a = min(1.0, len(u) / self.n_codes)
        self.u_mean = (1 - a) * self.u_mean + a * u.mean(0)
        self.u_var = (1 - a) * self.u_var + a * u.var(0, unbiased=False)
        return (u - self.u_mean) / (self.u_var.sqrt() + 1e-6)

    def propose(self):
        z = self.z_ctx.reshape(-1)
        u = self._codes()
        acts = self.vae.decode(z, u)
        if self.cfg.k_diverse:
            acts = torch.cat([acts, self.vae.decode(z, torch.randn(self.cfg.k_diverse, self.vae.d_u, device=z.device))])
        return acts, torch.full((len(acts), 1), float('nan'), device=z.device)

    def select(self, z, imagine_fn):
        self.z_ctx = z.reshape(1, -1)
        return super().select(z, imagine_fn)
