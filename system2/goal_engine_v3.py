"""System 2, v3: continuously driven reservoir + a readout that actually learns.

Changes vs v2 (goal_engine_v2.py), each aimed at a reason the pattern could not matter before:
  1. Continuous drive. Every candidate k has its own pattern stream position and its own reservoir state, and
     the reservoir is stepped on EVERY env step (drive()), not once per goal. An episode now feeds ~1000 pattern
     symbols per stream into the reservoir instead of ~10, so temporal structure (runs, self-similarity) can
     shape the reservoir state.
  2. Coherent streams. Candidate k always reads stream k (offset k * spread), so a committed goal comes from one
     contiguous stretch of the pattern; streams are not hopped between at every decision.
  3. A readout that can learn. v2 regressed W r onto normalise(W r) (its own output), so the error was ~0 and
     nothing was learned. v3 proposes a = normalise(W r) + sigma * eps and, after the goal ends, does
     advantage-weighted least squares (RLS) toward the executed a with weight max(0, gain/baseline - 1):
     only proposals that found more novelty than usual pull the readout. The readout persists across episodes.
  4. Pattern-only candidates by default (k_diverse=0), so EFE chooses among pattern-made proposals only and a
     pattern effect is not diluted by random candidates.
EFE-style choice (as v2): G = -nu * novelty(goal) - gamma * epistemic(goal) + beta * instability(goal).
Controls: stream=None -> reservoir driven by Gaussian noise of the same dimension ('none' = no pattern at all
uses i.i.d. random macro-actions instead of the reservoir).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .goal_engine import EnsembleDynamics, Reservoir
from .patterns import PatternStream


@dataclass
class V3Config:
    k: int = 16                    # candidates (= streams = reservoir copies)
    k_diverse: int = 0             # extra random macro-action candidates (0 = pattern only)
    horizon: int = 5               # top-level macro-steps imagined per goal
    action_dim: int = 4
    act_scale: float = 1.0
    sigma: float = 0.5             # proposal noise on the readout output (what learning can select on)
    spread: int = 997              # stream offset between candidates
    proj_dim: int = 16
    knn: int = 3
    nu_novel: float = 1.0
    gamma_epi: float = 0.5
    beta_unstable: float = 0.5
    learn_readout: bool = True
    readout_lr: float = 1.0        # scales the advantage weight (realised novelty of the executed goal)
    learn_imagined: bool = True    # also learn from all K imagined candidates at every decision
    imagined_lr: float = 0.3       # weight per unit of positive imagined-novelty advantage (z-score)
    ridge: float = 10.0            # RLS prior precision
    stall_patience: int = 2
    max_age: int = 6
    reach_frac: float = 0.35
    random_actions: bool = False   # 'none' control: no reservoir, i.i.d. Gaussian macro-actions
    device: str = 'cuda'


class GoalEngineV3:
    def __init__(self, d: int, stream: PatternStream | None, cfg: V3Config = V3Config(), seed: int = 0,
                 reservoir_n: int = 512):
        self.cfg, self.d, self.stream = cfg, d, stream
        dev = cfg.device
        g = torch.Generator().manual_seed(seed)
        self.in_p = stream.dim if stream is not None else 16
        self.res = Reservoir(self.in_p, d, n=reservoir_n, seed=seed).to(dev)
        out = cfg.horizon * cfg.action_dim
        self.W = (torch.randn(out, reservoir_n, generator=g) / np.sqrt(reservoir_n)).to(dev)
        self.P = torch.eye(reservoir_n, device=dev) / cfg.ridge
        self.proj = (torch.randn(d, cfg.proj_dim, generator=g) / np.sqrt(d)).to(dev)
        self.r = torch.zeros(cfg.k, reservoir_n, device=dev)
        self.pos = np.arange(cfg.k) * cfg.spread
        self.z_ctx = torch.zeros(1, d, device=dev)
        self.ens = EnsembleDynamics(d).to(dev)
        self.mem = torch.empty(0, cfg.proj_dim, device=dev)
        self.goal = None
        self._gains: list[float] = []
        self.noise_rng = np.random.default_rng(seed + 7)
        self.stats = dict(selections=0, updates=0, update_weight=0.0)

    # ---------------------------------------------------------------- episode / drive
    def reset(self, episode_offset: int = 0):
        self.pos = np.arange(self.cfg.k) * self.cfg.spread + episode_offset
        self.r.zero_(); self.goal = None
        self.mem = self.mem[:0]

    def _inputs(self) -> torch.Tensor:
        if self.stream is not None:
            p = np.stack([self.stream.peek(int(o) - self.stream.pos) for o in self.pos])
        else:
            p = self.noise_rng.normal(size=(self.cfg.k, self.in_p)).astype(np.float32)
        self.pos += 1
        return torch.from_numpy(p).to(self.cfg.device)

    def drive(self, z: torch.Tensor | None = None):
        """Advance every candidate's reservoir by one pattern symbol. Call once per env step."""
        if self.cfg.random_actions:
            return
        if z is not None:
            self.z_ctx = z.reshape(1, -1)
        self.r = self.res.step(self.r, self._inputs(), self.z_ctx.expand(self.cfg.k, -1))

    # ---------------------------------------------------------------- novelty memory
    def observe(self, z):
        self.mem = torch.cat([self.mem, z.reshape(1, -1) @ self.proj])[-4000:]

    def novelty(self, goals):
        if len(self.mem) < self.cfg.knn:
            return torch.zeros(len(goals), device=goals.device)
        return torch.cdist(goals @ self.proj, self.mem).topk(self.cfg.knn, largest=False).values.mean(-1)

    # ---------------------------------------------------------------- proposals and choice
    def propose(self):
        c, dev = self.cfg, self.cfg.device
        H, A = c.horizon, c.action_dim
        if c.random_actions:
            a = torch.randn(c.k + c.k_diverse, H * A, device=dev)
            return c.act_scale * a.reshape(-1, H, A), torch.full((len(a), self.r.shape[1]), float('nan'), device=dev)
        mean = self.r @ self.W.T
        mean = (mean - mean.mean(-1, keepdim=True)) / (mean.std(-1, keepdim=True) + 1e-6)
        a = mean + c.sigma * torch.randn_like(mean)
        R = self.r.clone()
        if c.k_diverse:
            a = torch.cat([a, torch.randn(c.k_diverse, H * A, device=dev)])
            R = torch.cat([R, torch.full((c.k_diverse, R.shape[1]), float('nan'), device=dev)])
        return c.act_scale * a.reshape(-1, H, A), R

    @staticmethod
    def _zs(x):
        return (x - x.mean()) / (x.std() + 1e-6)

    def select(self, z, imagine_fn):
        c = self.cfg
        acts, R = self.propose()
        goals = imagine_fn(z, acts)
        unstable = (imagine_fn(z, acts + 0.1 * torch.randn_like(acts)) - goals).norm(dim=-1)
        nov = self.novelty(goals)
        epi = self.ens.disagreement(z.reshape(1, -1), goals)
        G = -c.nu_novel * self._zs(nov) - c.gamma_epi * self._zs(epi) + c.beta_unstable * self._zs(unstable)
        k = int(torch.argmin(G))
        d0 = float((goals[k] - z).norm())
        self.goal = dict(z0=z.detach(), goal=goals[k].detach(), act=acts[k].detach() / c.act_scale, r=R[k].detach(),
                         d0=d0, best=d0, stall=0, age=0, mem_len=len(self.mem))
        self.stats['selections'] += 1
        if c.learn_readout and c.learn_imagined and len(self.mem) >= c.knn:
            adv = self._zs(nov)
            for i in range(len(acts)):
                if adv[i] > 0 and bool(torch.isfinite(R[i]).all()):
                    self._rls(R[i], acts[i].reshape(-1) / c.act_scale, min(3.0, c.imagined_lr * float(adv[i])))
                    self.stats['imagined_updates'] = self.stats.get('imagined_updates', 0) + 1
        return goals[k], dict(G=float(G[k]), nov=float(nov[k]), epi=float(epi[k]), unstable=float(unstable[k]), k=k)

    def should_reselect(self, z) -> bool:
        g = self.goal
        if g is None:
            return True
        g['age'] += 1
        dist = float((g['goal'] - z).norm())
        if dist < g['best'] - 0.02 * g['d0']:
            g['best'], g['stall'] = dist, 0
        else:
            g['stall'] += 1
        return dist < self.cfg.reach_frac * g['d0'] or g['stall'] >= self.cfg.stall_patience or g['age'] >= self.cfg.max_age

    # ---------------------------------------------------------------- learning
    def finish_goal(self, z_reached):
        g = self.goal
        if g is None:
            return
        self.ens.add(g['z0'], g['goal'], z_reached.detach())
        self.ens.fit(steps=20, device=z_reached.device)
        old = self.mem[:g['mem_len']]
        gain = float(torch.cdist(z_reached.reshape(1, -1) @ self.proj, old).topk(self.cfg.knn, largest=False)
                     .values.mean()) if len(old) >= self.cfg.knn else None
        if gain is not None:
            base = float(np.median(self._gains[-50:])) if len(self._gains) >= 5 else None
            self._gains.append(gain)
            r = g['r']
            if self.cfg.learn_readout and base is not None and bool(torch.isfinite(r).all()):
                w = self.cfg.readout_lr * max(0.0, gain / (base + 1e-6) - 1.0)
                w = min(w, 3.0)
                if w > 0:
                    self._rls(r, g['act'].reshape(-1), w)
                    self.stats['updates'] += 1; self.stats['update_weight'] += w
        self.goal = None

    def _rls(self, r, y, w):
        """Weighted recursive least squares step: pull W r toward y with sample weight w."""
        Pr = self.P @ r
        kk = Pr / (1.0 / w + r @ Pr)
        self.W += torch.outer(y - self.W @ r, kk)
        self.P -= torch.outer(kk, Pr)
