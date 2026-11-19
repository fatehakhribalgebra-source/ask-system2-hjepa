"""The agent's probabilistic generative model in the H-JEPA top-level latent ("the environment is the latent").

GaussEnsemble: M diagonal-Gaussian models p_m(z' | z, a) = N(z + mu_m(z, a), diag sigma_m^2(z, a)), in a
standardised latent space. Trained from scratch on the agent's OWN transitions (online), optionally warm-started
from real data. H-JEPA's predictor is deterministic; this model carries the uncertainty that EFE needs.

MacroActionTracker: recovers the TRUE level-3 macro-action the agent just executed (and the latents around it)
by pushing the last 40 env steps of executed torques + observations through the frozen H-JEPA exactly as in
training (HJEPA.encode_hierarchical, no crops) -> exact (z3_t, a3_t, z3_t+1) transitions for online learning.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn

LOG2PIE = math.log(2 * math.pi * math.e)


class GaussEnsemble(nn.Module):
    def __init__(self, d: int, action_dim: int = 4, m: int = 5, hidden: int = 512, lr: float = 1e-3,
                 z_mean=None, z_std=None, proj=None):
        """proj: optional (k, D) whitening projection (PCA of real latents scaled by 1/sqrt(eigenvalue)); the model
        then observes y = proj @ (z - z_mean), k-dim with ~N(0, I) marginals over real data. Without it, per-dim
        standardisation of all D dims (which over-counts correlated dimensions in the entropy terms)."""
        super().__init__()
        self.m = m
        D = d
        self.register_buffer('z_mean', torch.zeros(D) if z_mean is None else torch.as_tensor(z_mean).float())
        self.register_buffer('z_std', torch.ones(D) if z_std is None else torch.as_tensor(z_std).float())
        self.register_buffer('proj', torch.zeros(0, D) if proj is None else torch.as_tensor(proj).float())
        d = len(self.proj) if proj is not None else D
        self.d = d
        self.heads = nn.ModuleList([nn.Sequential(
            nn.Linear(d + action_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * d)) for _ in range(m)])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.buf_z, self.buf_a, self.buf_zn = [], [], []

    # all internal computation in standardised latent units
    def n(self, z):
        if len(self.proj):
            return (z - self.z_mean) @ self.proj.T
        return (z - self.z_mean) / self.z_std

    @staticmethod
    def pca_whitener(Z, k: int = 32):
        """Z (N, D) real latents -> (mean (D,), proj (k, D)) so that proj @ (z - mean) is whitened."""
        Z = torch.as_tensor(Z, dtype=torch.float64)
        mu = Z.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((Z - mu).T))
        ev, V = ev.flip(0)[:k], V.flip(1)[:, :k]
        return mu.float(), (V / ev.sqrt()).T.float()

    def step_dist(self, zn, a):
        """zn: (M, K, d) standardised states per head, a: (K, A) -> mean, var (M, K, d) of next standardised state."""
        mus, vs = [], []
        for i, h in enumerate(self.heads):
            o = h(torch.cat([zn[i], a], -1))
            mu, lv = o.chunk(2, -1)
            mus.append(zn[i] + mu); vs.append(lv.clamp(-8, 2).exp())
        return torch.stack(mus), torch.stack(vs)

    @torch.no_grad()
    def rollout_terms(self, z, acts):
        """Moment-propagated rollout of K policies acts (K, H, A) from z (D,). Returns per-step (K, H) tensors:
        info_gain (nats; mixture entropy - mean component entropy), ambiguity (mean component entropy),
        and the moment-matched predictive mean/var (K, H, d) for the risk term."""
        K, H, _ = acts.shape
        zn = self.n(z.reshape(1, 1, -1)).expand(self.m, K, -1)
        ig, amb, means, vars_ = [], [], [], []
        for t in range(H):
            mu, var = self.step_dist(zn, acts[:, t])
            mbar = mu.mean(0)
            vtot = var.mean(0) + mu.var(0, unbiased=False)                       # moment-matched mixture
            h_mix = 0.5 * (LOG2PIE + vtot.log()).sum(-1)
            h_comp = (0.5 * (LOG2PIE + var.log()).sum(-1)).mean(0)
            ig.append(h_mix - h_comp); amb.append(h_comp); means.append(mbar); vars_.append(vtot)
            zn = mu                                                               # each head follows its own mean
        return torch.stack(ig, 1), torch.stack(amb, 1), torch.stack(means, 1), torch.stack(vars_, 1)

    def add(self, z, a, zn):
        self.buf_z.append(z.detach().float().cpu()); self.buf_a.append(a.detach().float().cpu())
        self.buf_zn.append(zn.detach().float().cpu())

    def fit(self, steps: int = 30, batch: int = 128, device='cuda'):
        if len(self.buf_z) < 4:
            return None
        Z = self.n(torch.stack(self.buf_z).to(device)); A = torch.stack(self.buf_a).to(device)
        ZN = self.n(torch.stack(self.buf_zn).to(device))
        loss_v = None
        for _ in range(steps):
            loss = 0.0
            for h in self.heads:                                                  # independent bootstrap per head
                idx = torch.randint(len(Z), (min(batch, len(Z)),), device=device)
                o = h(torch.cat([Z[idx], A[idx]], -1))
                mu, lv = o.chunk(2, -1)
                lv = lv.clamp(-8, 2)
                loss = loss + (0.5 * ((ZN[idx] - Z[idx] - mu) ** 2 / lv.exp() + lv)).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            loss_v = float(loss) / self.m
        return loss_v


class MacroActionTracker:
    """Keeps the last 40 env steps and, every 20 steps, returns the exact (z3_t, a3_t, z3_t+1) transition."""

    def __init__(self, model, normalizer_stats: dict, device='cuda', fs: int = 5, span: int = 40):
        self.model, self.dev, self.fs, self.span = model, device, fs, span
        self.am = np.asarray(normalizer_stats['action']['mean'], np.float32).reshape(-1)
        self.asd = np.asarray(normalizer_stats['action']['std'], np.float32).reshape(-1)
        self.pm = np.asarray(normalizer_stats['proprio']['mean'], np.float32).reshape(-1)
        self.psd = np.asarray(normalizer_stats['proprio']['std'], np.float32).reshape(-1)
        self.buf = deque(maxlen=span)
        self.captured = []
        self._hook = model.jepas[-1].action_embed.register_forward_hook(lambda m, i, o: self.captured.append(i[0]))
        self._saved_tl = [getattr(j, 'target_length', None) for j in model.jepas]

    def reset(self):
        self.buf.clear()

    def push(self, pixels, qpos, qvel, action):
        self.buf.append((np.asarray(pixels, np.uint8).reshape(64, 64, 3), np.asarray(qpos).reshape(-1),
                         np.asarray(qvel).reshape(-1), np.asarray(action, np.float32).reshape(-1)))

    @torch.no_grad()
    def transition(self):
        if len(self.buf) < self.span:
            return None
        b = list(self.buf)
        idx = list(range(0, self.span, self.fs))                                   # 8 level-1 frames
        px = torch.from_numpy(np.stack([b[i][0] for i in idx])).float().div(255).permute(0, 3, 1, 2)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        px = ((px - mean) / std)[None].to(self.dev)
        prop = np.stack([np.concatenate([b[i][1][2:], b[i][2]]) for i in idx]).astype(np.float32)
        prop = (prop - self.pm) / self.psd
        acts = np.stack([((np.stack([b[i + k][3] for k in range(self.fs)]) - self.am) / self.asd).reshape(-1) for i in idx])
        info = {'pixels_level1': px, 'proprio_level1': torch.from_numpy(prop)[None].to(self.dev),
                'action_level1': torch.from_numpy(acts.astype(np.float32))[None].to(self.dev)}
        for j in self.model.jepas:
            j.target_length = None
        self.captured.clear()
        try:
            out = self.model.encode_hierarchical(info, sparse_level1_encode=False)
        finally:
            for j, tl in zip(self.model.jepas, self._saved_tl):
                j.target_length = tl
        z3 = out[f'embed_{self.model.num_levels}'][0]
        a3 = self.captured[-1][0]
        if len(z3) < 2 or len(a3) < 1:
            return None
        return z3[0], a3[0], z3[1]
