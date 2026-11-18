"""System 2, v2 ("clever"): pattern-seeded, learning, committed goal generation over a frozen H-JEPA.

Differences from v1 (goal_engine.py), each motivated by the v1 ablations:
  1. Commitment: a goal is kept while H-JEPA keeps making latent progress toward it; re-selection only when
     progress stalls, the goal is reached, or a max age passes (v1 replaced goals on a fixed clock).
  2. Place-like novelty: kNN novelty in a fixed random projection (dim 16) of the top-level latent, against an
     episodic memory, plus a bonus for imagined goals far from every remembered state (frontier).
  3. Mixed proposals: K_p candidates from the pattern -> reservoir -> readout pathway and K_d "diverse" candidates
     (random macro-actions); EFE picks among all of them, so the pattern is a prior, not a dictator.
  4. Learning readout: after each committed goal, the readout W_act is pulled (reward-weighted, RLS-style
     least squares) toward the macro-actions that produced realised novelty, conditioned on the reservoir state
     that proposed them. Fixed pattern + learned readout = the "pattern + learning" combination.
EFE-style score:  G = -nu * novelty(goal) - gamma * epistemic(goal) + beta * (imagination spread / instability)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .goal_engine import EnsembleDynamics, Reservoir
from .patterns import PatternStream


@dataclass
class V2Config:
    k_pattern: int = 12
    k_diverse: int = 12
    horizon: int = 3
    action_dim: int = 4
    act_scale: float = 1.0
    proj_dim: int = 16
    knn: int = 3
    nu_novel: float = 1.0
    gamma_epi: float = 0.5
    beta_unstable: float = 0.5
    readout_lr: float = 0.5
    learn_readout: bool = True
    use_pattern: bool = True          # False -> only diverse candidates (pattern-free control)
    stall_patience: int = 2           # checks without progress before re-selecting
    max_age: int = 6                  # checks (each check = one replanning window)
    reach_frac: float = 0.35          # goal counts as reached when dist < reach_frac * initial dist
    device: str = 'cuda'


class GoalEngineV2:
    def __init__(self, d: int, stream: PatternStream | None, cfg: V2Config = V2Config(), seed: int = 0,
                 reservoir_n: int = 512):
        self.cfg, self.d, self.stream = cfg, d, stream
        dev = cfg.device
        g = torch.Generator().manual_seed(seed)
        in_p = stream.dim if stream is not None else 4
        self.res = Reservoir(in_p, d, n=reservoir_n, seed=seed).to(dev)
        out = cfg.horizon * cfg.action_dim
        self.W_act = (torch.randn(out, reservoir_n, generator=g) / np.sqrt(reservoir_n)).to(dev)
        self.P = torch.eye(reservoir_n, device=dev)                       # RLS inverse covariance
        self.proj = (torch.randn(d, cfg.proj_dim, generator=g) / np.sqrt(d)).to(dev)
        self.r = torch.zeros(cfg.k_pattern, reservoir_n, device=dev)
        self.ens = EnsembleDynamics(d).to(dev)
        self.mem = torch.empty(0, cfg.proj_dim, device=dev)
        self.goal = None
        self.stats = dict(selections=0, from_pattern=0)
        self._gains: list[float] = []

    # ---------------------------------------------------------------- memory / novelty
    def reset(self, episode_offset: int = 0):
        if self.stream is not None:
            self.stream.reset(episode_offset)
        self.r.zero_(); self.goal = None
        self.mem = self.mem[:0]

    def observe(self, z: torch.Tensor):
        self.mem = torch.cat([self.mem, (z.reshape(1, -1) @ self.proj)])[-4000:]

    def novelty(self, goals: torch.Tensor) -> torch.Tensor:
        if len(self.mem) < self.cfg.knn:
            return torch.zeros(len(goals), device=goals.device)
        d = torch.cdist(goals @ self.proj, self.mem)
        return d.topk(self.cfg.knn, largest=False).values.mean(-1)

    # ---------------------------------------------------------------- proposals
    def _pattern_actions(self, z):
        c, dev = self.cfg, self.cfg.device
        K = c.k_pattern
        p = torch.from_numpy(np.stack([self.stream.peek(i * 997) for i in range(K)])).to(dev)
        self.stream.next()
        self.r = self.res.step(self.r, p, z.reshape(1, -1).expand(K, -1))
        a = self.r @ self.W_act.T
        a = (a - a.mean(-1, keepdim=True)) / (a.std(-1, keepdim=True) + 1e-6)
        return c.act_scale * a.reshape(K, c.horizon, c.action_dim), self.r.clone()

    def propose(self, z):
        c = self.cfg
        acts, R = [], []
        if c.use_pattern and self.stream is not None:
            a, r = self._pattern_actions(z); acts.append(a); R.append(r)
        kd = c.k_diverse if (c.use_pattern and self.stream is not None) else c.k_diverse + c.k_pattern
        acts.append(c.act_scale * torch.randn(kd, c.horizon, c.action_dim, device=c.device))
        R.append(torch.full((kd, self.r.shape[1]), float('nan'), device=c.device))
        return torch.cat(acts), torch.cat(R)

    # ---------------------------------------------------------------- selection
    @staticmethod
    def _z(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    def select(self, z, imagine_fn):
        c = self.cfg
        acts, R = self.propose(z)
        goals = imagine_fn(z, acts)
        # instability: how much the imagined endpoint moves under a small action perturbation
        goals_p = imagine_fn(z, acts + 0.1 * torch.randn_like(acts))
        unstable = (goals_p - goals).norm(dim=-1)
        nov = self.novelty(goals)
        epi = self.ens.disagreement(z.reshape(1, -1), goals)
        G = -c.nu_novel * self._z(nov) - c.gamma_epi * self._z(epi) + c.beta_unstable * self._z(unstable)
        k = int(torch.argmin(G))
        self.goal = dict(z0=z.detach(), goal=goals[k].detach(), act=acts[k].detach(), r=R[k].detach(),
                         d0=float((goals[k] - z).norm()), best=float((goals[k] - z).norm()), stall=0, age=0,
                         mem_len=len(self.mem), from_pattern=bool(torch.isfinite(R[k]).all()))
        self.stats['selections'] += 1; self.stats['from_pattern'] += int(self.goal['from_pattern'])
        return goals[k], dict(G=float(G[k]), nov=float(nov[k]), epi=float(epi[k]), unstable=float(unstable[k]),
                              from_pattern=self.goal['from_pattern'])

    # ---------------------------------------------------------------- commitment
    def should_reselect(self, z) -> bool:
        if self.goal is None:
            return True
        g = self.goal
        g['age'] += 1
        dist = float((g['goal'] - z).norm())
        if dist < g['best'] - 0.02 * g['d0']:
            g['best'], g['stall'] = dist, 0
        else:
            g['stall'] += 1
        return dist < self.cfg.reach_frac * g['d0'] or g['stall'] >= self.cfg.stall_patience or g['age'] >= self.cfg.max_age

    # ---------------------------------------------------------------- learning
    def finish_goal(self, z_reached):
        """Call when a committed goal ends. Updates ensemble and (pattern goals only) the readout."""
        g = self.goal
        if g is None:
            return
        self.ens.add(g['z0'], g['goal'], z_reached.detach())
        self.ens.fit(steps=20, device=z_reached.device)
        if self.cfg.learn_readout and g['from_pattern']:
            # realised novelty of the committed segment: novelty of where we ended up vs memory before the goal
            old = self.mem[:g['mem_len']]
            if len(old) >= self.cfg.knn:
                d = torch.cdist((z_reached.reshape(1, -1) @ self.proj), old)
                gain = float(d.topk(self.cfg.knn, largest=False).values.mean())
            else:
                gain = 1.0
            w = float(np.clip(gain / (1e-6 + self._typical_gain()), 0.0, 3.0)) * self.cfg.readout_lr
            if w > 0:
                r, y = g['r'], g['act'].reshape(-1)
                Pr = self.P @ r
                k = Pr / (1.0 / w + r @ Pr)
                self.W_act += torch.outer(y - self.W_act @ r, k)
                self.P -= torch.outer(k, Pr)
            self._gains.append(gain)
        self.goal = None

    def _typical_gain(self):
        return float(np.median(self._gains[-50:])) if self._gains else 1.0
