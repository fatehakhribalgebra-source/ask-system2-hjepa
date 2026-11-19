"""System 2, baseline C: no generator. Cross-entropy method directly on the EFE score.

Population of K macro-action sequences ~ N(mu, sigma), initial sigma = the per-dimension std of TRUE level-3
macro-actions (from the VAE's stats, or 0.65). Each round: imagine with H-JEPA L3, score with the same EFE
terms as v3/G (novelty, ensemble disagreement, instability), refit mu/sigma to the top `elites`. The best
candidate of the last round is committed to, exactly as in v3/G. This is the "is a generator needed at all?"
ceiling for a given imagination budget (rounds x K imagined rollouts per decision).
"""
from __future__ import annotations

import torch

from .goal_engine_v3 import GoalEngineV3, V3Config


class GoalEngineCEM(GoalEngineV3):
    def __init__(self, d: int, cfg: V3Config = V3Config(), seed: int = 0, rounds: int = 3, elites: int = 4,
                 init_std: torch.Tensor | float = 0.65):
        cfg.learn_readout = False
        cfg.random_actions = True
        super().__init__(d, None, cfg, seed=seed, reservoir_n=8)
        self.rounds, self.elites = rounds, elites
        self.init_std = torch.as_tensor(init_std, dtype=torch.float32, device=cfg.device).reshape(-1)
        self.gen = torch.Generator(device=cfg.device).manual_seed(seed + 23)

    def _score(self, z, acts, imagine_fn):
        c = self.cfg
        goals = imagine_fn(z, acts)
        unstable = (imagine_fn(z, acts + 0.1 * torch.randn_like(acts)) - goals).norm(dim=-1)
        nov = self.novelty(goals)
        epi = self.ens.disagreement(z.reshape(1, -1), goals)
        G = -c.nu_novel * self._zs(nov) - c.gamma_epi * self._zs(epi) + c.beta_unstable * self._zs(unstable)
        return goals, G, nov, epi, unstable

    def select(self, z, imagine_fn):
        c = self.cfg
        H, A = c.horizon, c.action_dim
        mu = torch.zeros(H * A, device=c.device)
        std = self.init_std.repeat(H * A // len(self.init_std)) if len(self.init_std) < H * A else self.init_std.clone()
        for _ in range(self.rounds):
            acts = (mu + std * torch.randn(c.k, H * A, device=c.device, generator=self.gen)).reshape(c.k, H, A)
            goals, G, nov, epi, unstable = self._score(z, acts, imagine_fn)
            el = acts.reshape(c.k, -1)[torch.argsort(G)[:self.elites]]
            mu, std = el.mean(0), el.std(0) + 1e-3
        k = int(torch.argmin(G))
        d0 = float((goals[k] - z).norm())
        self.goal = dict(z0=z.detach(), goal=goals[k].detach(), act=acts[k].detach(), r=torch.full((1,), float('nan')),
                         d0=d0, best=d0, stall=0, age=0, mem_len=len(self.mem))
        self.stats['selections'] += 1
        return goals[k], dict(G=float(G[k]), nov=float(nov[k]), epi=float(epi[k]), unstable=float(unstable[k]), k=k)
