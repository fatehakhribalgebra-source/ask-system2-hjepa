"""System 2: pattern-driven reservoir goal proposer + EFE-style scorer.

World-model agnostic: it needs only
  - the current top-level latent z (D,)
  - reach_fn(z, goals[K, D]) -> (reach_err[K], pred_traj[K, H, D] or None)
  - an epistemic ensemble (EnsembleDynamics) trained on observed top-level transitions
and optionally a preference prior mu_c (D,).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .patterns import PatternStream


# ---------------------------------------------------------------- reservoir
class Reservoir(nn.Module):
    """Fixed leaky echo-state network. Nothing in here is trained."""

    def __init__(self, in_p: int, in_z: int, n: int = 512, rho: float = 0.95, leak: float = 0.3,
                 density: float = 0.05, p_scale: float = 1.0, z_scale: float = 0.1, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(n, n, generator=g) * (torch.rand(n, n, generator=g) < density)
        w = w / torch.linalg.eigvals(w).abs().max() * rho
        self.register_buffer('W', w)
        self.register_buffer('W_p', torch.randn(n, in_p, generator=g) * p_scale / np.sqrt(in_p))
        self.register_buffer('W_z', torch.randn(n, in_z, generator=g) * z_scale / np.sqrt(in_z))
        self.leak = leak
        self.n = n

    def step(self, r: torch.Tensor, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """r: (K, n), p: (K, in_p), z: (K, in_z) -> (K, n)"""
        pre = r @ self.W.T + p @ self.W_p.T + z @ self.W_z.T
        return (1 - self.leak) * r + self.leak * torch.tanh(pre)


# ---------------------------------------------------------------- epistemic ensemble
class EnsembleDynamics(nn.Module):
    """M small MLPs predicting the achieved top-level displacement from (z, goal).

    Trained online with bootstrap masks; disagreement = epistemic uncertainty about
    what pursuing a goal actually does from here.
    """

    def __init__(self, d: int, m: int = 5, hidden: int = 256, lr: float = 1e-3):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Linear(hidden, d)) for _ in range(m)])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.buf_x: list[torch.Tensor] = []
        self.buf_y: list[torch.Tensor] = []

    def forward(self, z: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:  # -> (M, K, d)
        x = torch.cat([z.expand_as(goal), goal - z], dim=-1)
        return torch.stack([h(x) for h in self.heads])

    def disagreement(self, z, goal) -> torch.Tensor:
        with torch.no_grad():
            p = self(z, goal)
        return p.var(0).sum(-1)

    def add(self, z, goal, z_next) -> None:
        self.buf_x.append(torch.cat([z, goal - z]).detach().cpu())
        self.buf_y.append((z_next - z).detach().cpu())

    def fit(self, steps: int = 20, batch: int = 128, device='cpu') -> float | None:
        if len(self.buf_x) < 8:
            return None
        X, Y = torch.stack(self.buf_x).to(device), torch.stack(self.buf_y).to(device)
        loss_v = 0.0
        for _ in range(steps):
            loss = 0.0
            for h in self.heads:                          # independent bootstrap per head
                idx = torch.randint(len(X), (min(batch, len(X)),), device=device)
                loss = loss + ((h(X[idx]) - Y[idx]) ** 2).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            loss_v = float(loss) / len(self.heads)
        return loss_v


# ---------------------------------------------------------------- engine
@dataclass
class EngineConfig:
    k: int = 16                 # candidates per decision
    step_norm: float = 1.0      # max ||goal - z|| (calibrate to typical top-level displacement)
    beta_reach: float = 1.0
    gamma_epi: float = 1.0
    nu_novel: float = 1.0
    alpha_pref: float = 0.0     # 0 = pure exploration
    knn: int = 3
    memory: int = 5000
    ridge: float = 1.0
    learn_readout: bool = True
    seed_spread: int = 997      # stream offset between candidates
    device: str = 'cuda'


class GoalEngine:
    def __init__(self, d: int, stream: PatternStream, cfg: EngineConfig = EngineConfig(),
                 reservoir_n: int = 512, seed: int = 0):
        self.cfg, self.d, self.stream = cfg, d, stream
        dev = cfg.device
        self.res = Reservoir(stream.dim, d, n=reservoir_n, seed=seed).to(dev)
        g = torch.Generator().manual_seed(seed + 1)
        # readout: random init, optionally refit by recursive least squares (hindsight)
        self.W_out = (torch.randn(d, reservoir_n, generator=g) / np.sqrt(reservoir_n)).to(dev)
        self.P = torch.eye(reservoir_n, device=dev) / cfg.ridge     # RLS inverse covariance
        self.r = torch.zeros(cfg.k, reservoir_n, device=dev)
        self.ens = EnsembleDynamics(d).to(dev)
        self.mem = torch.empty(0, d, device=dev)
        self.mu_c: torch.Tensor | None = None
        self.last = None

    # -- episode control
    def reset(self, episode_offset: int = 0) -> None:
        self.stream.reset(episode_offset)
        self.r.zero_()
        self.mem = self.mem[-self.cfg.memory:]

    def observe(self, z: torch.Tensor) -> None:
        self.mem = torch.cat([self.mem, z.detach().reshape(1, -1)])[-self.cfg.memory:]

    # -- proposal
    def propose(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        K, dev = self.cfg.k, self.cfg.device
        p = torch.from_numpy(np.stack([self.stream.peek(i * self.cfg.seed_spread) for i in range(K)])).to(dev)
        self.stream.next()
        zk = z.reshape(1, -1).expand(K, -1)
        self.r = self.res.step(self.r, p, zk)
        delta = self.r @ self.W_out.T
        n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta = delta * (self.cfg.step_norm * torch.tanh(n / self.cfg.step_norm) / n)
        return zk + delta, self.r.clone()

    # -- scoring
    def novelty(self, goals: torch.Tensor) -> torch.Tensor:
        if len(self.mem) < self.cfg.knn:
            return torch.zeros(len(goals), device=goals.device)
        d = torch.cdist(goals, self.mem)
        return d.topk(self.cfg.knn, largest=False).values.mean(-1)

    def score(self, z, goals, reach_err) -> dict:
        c = self.cfg
        epi = self.ens.disagreement(z.reshape(1, -1), goals)
        nov = self.novelty(goals)
        pref = ((goals - self.mu_c) ** 2).sum(-1) if (self.mu_c is not None and c.alpha_pref > 0) \
            else torch.zeros_like(nov)
        norm = lambda x: (x - x.mean()) / (x.std() + 1e-6)          # put terms on one scale
        G = c.beta_reach * norm(reach_err) - c.gamma_epi * norm(epi) - c.nu_novel * norm(nov) \
            + c.alpha_pref * norm(pref)
        return dict(G=G, reach=reach_err, epi=epi, nov=nov, pref=pref)

    def select(self, z: torch.Tensor, reach_fn) -> tuple[torch.Tensor, dict]:
        goals, R = self.propose(z)
        reach_err = reach_fn(z, goals)
        s = self.score(z, goals, reach_err)
        k = int(torch.argmin(s['G']))
        self.last = dict(z=z.detach(), goal=goals[k].detach(), r=R[k].detach(), k=k)
        return goals[k], {key: v.detach().cpu() for key, v in s.items()} | {'k': k}

    # -- imagination mode: pattern -> reservoir -> abstract macro-actions -> F(L) rollout -> goal
    def init_imagination(self, horizon: int, action_dim: int, act_scale: float = 1.0) -> None:
        g = torch.Generator().manual_seed(12345)
        n = self.res.n
        self.H_im, self.adim, self.act_scale = horizon, action_dim, act_scale
        self.W_act = (torch.randn(horizon * action_dim, n, generator=g) / np.sqrt(n)).to(self.cfg.device)

    def propose_actions(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        K, dev = self.cfg.k, self.cfg.device
        p = torch.from_numpy(np.stack([self.stream.peek(i * self.cfg.seed_spread) for i in range(K)])).to(dev)
        self.stream.next()
        zk = z.reshape(1, -1).expand(K, -1)
        self.r = self.res.step(self.r, p, zk)
        a = self.r @ self.W_act.T
        a = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-6)   # unit scale per candidate
        return self.act_scale * a.reshape(K, self.H_im, self.adim), self.r.clone()

    def select_imagined(self, z: torch.Tensor, imagine_fn, use_efe: bool = True):
        """imagine_fn(z, acts[K,H,A]) -> predicted top-level states [K, D] (on the model's manifold)."""
        acts, R = self.propose_actions(z)
        goals = imagine_fn(z, acts)
        if use_efe:
            s = self.score(z, goals, torch.zeros(len(goals), device=goals.device))
            k = int(torch.argmin(s['G']))
        else:
            s = {key: torch.zeros(len(goals)) for key in ('G', 'reach', 'epi', 'nov', 'pref')}
            k = 0
        self.last = dict(z=z.detach(), goal=goals[k].detach(), r=R[k].detach(), k=k)
        return goals[k], {key: v.detach().cpu() for key, v in s.items()} | {'k': k}

    # -- learning after the goal was pursued
    def update(self, z_reached: torch.Tensor, novelty_gain: float | None = None) -> None:
        if self.last is None:
            return
        z0, goal, r = self.last['z'], self.last['goal'], self.last['r']
        self.ens.add(z0, goal, z_reached.detach())
        if self.cfg.learn_readout and getattr(self, 'W_act', None) is None:
            # hindsight RLS: make this reservoir state propose the displacement actually achieved
            w = 1.0 if novelty_gain is None else float(np.clip(novelty_gain, 0.1, 10.0))
            y = (z_reached.detach() - z0)
            Pr = self.P @ r
            gain = Pr / (1.0 / w + r @ Pr)
            err = y - self.W_out @ r
            self.W_out += torch.outer(err, gain)
            self.P -= torch.outer(gain, Pr)
