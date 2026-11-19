"""Conditional VAE over top-level macro-action sequences: p(a_1..a_H | z) = int p(a | z, u) N(u; 0, I) du.

Trained on TRUE level-3 macro-actions of real ant trajectories (experiments/build_macro_action_data.py), so its
samples lie on the action distribution the frozen H-JEPA predictor was trained with. System 2 replaces the
Gaussian code u by a code made from a pattern stream (goal_engine_gen.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ActionVAE(nn.Module):
    def __init__(self, d_z: int, horizon: int = 5, action_dim: int = 4, d_u: int = 8, hidden: int = 512):
        super().__init__()
        self.H, self.A, self.d_u = horizon, action_dim, d_u
        out = horizon * action_dim
        self.register_buffer('z_mean', torch.zeros(d_z)); self.register_buffer('z_std', torch.ones(d_z))
        self.register_buffer('a_mean', torch.zeros(out)); self.register_buffer('a_std', torch.ones(out))
        self.cond = nn.Sequential(nn.Linear(d_z, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.enc = nn.Sequential(nn.Linear(hidden + out, hidden), nn.SiLU(), nn.Linear(hidden, 2 * d_u))
        self.dec = nn.Sequential(nn.Linear(hidden + d_u, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, out))

    def set_stats(self, z, a):
        self.z_mean.copy_(z.mean(0)); self.z_std.copy_(z.std(0) + 1e-6)
        a = a.reshape(len(a), -1)
        self.a_mean.copy_(a.mean(0)); self.a_std.copy_(a.std(0) + 1e-6)

    def _c(self, z):
        return self.cond((z - self.z_mean) / self.z_std)

    def forward(self, z, a):
        c = self._c(z)
        an = (a.reshape(len(a), -1) - self.a_mean) / self.a_std
        mu, logvar = self.enc(torch.cat([c, an], -1)).chunk(2, -1)
        u = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        rec = self.dec(torch.cat([c, u], -1))
        rec_loss = ((rec - an) ** 2).sum(-1).mean()
        kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).sum(-1).mean()
        return rec_loss, kl

    @torch.no_grad()
    def log_prior(self, z, a):
        """ELBO-based log p(a | z) (nats, in standardised action units): the policy prior ln E(pi)."""
        z = z.reshape(-1, z.shape[-1]).expand(len(a), -1)
        c = self._c(z)
        an = (a.reshape(len(a), -1) - self.a_mean) / self.a_std
        mu, logvar = self.enc(torch.cat([c, an], -1)).chunk(2, -1)
        rec = self.dec(torch.cat([c, mu], -1))
        kl = (0.5 * (mu ** 2 + logvar.exp() - 1 - logvar)).sum(-1)
        return -0.5 * ((rec - an) ** 2).sum(-1) - kl

    @torch.no_grad()
    def decode(self, z, u):
        """z: (K, D) or (D,), u: (K, d_u) -> macro-actions (K, H, A) in the model's action units."""
        z = z.reshape(-1, z.shape[-1]).expand(len(u), -1)
        an = self.dec(torch.cat([self._c(z), u], -1))
        return (an * self.a_std + self.a_mean).reshape(len(u), self.H, self.A)
