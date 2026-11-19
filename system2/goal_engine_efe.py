"""System 2 with real expected free energy, the H-JEPA top-level latent treated as the environment.

Generative model (latent_dynamics.GaussEnsemble): p_m(z' | z, a), M diagonal Gaussians, learned online from the
agent's own exact (z3, a3, z3') transitions. Policies pi = 5-step macro-action sequences, proposed by the
pattern-driven VAE (generator G). For each policy, moment-propagated rollout of the ensemble gives per step tau:
    info gain_tau  = H[mixture] - mean_m H[p_m]          (parameter information gain, closed form for Gaussians)
    ambiguity_tau  = mean_m H[p_m]                         (expected entropy of outcomes: the noisy-TV penalty)
    risk_tau       = KL( q(z_tau | pi) || p_C(z) )         (p_C = N(0, I) in data-standardised latent units:
                                                            the distribution of real latents, i.e. stay on manifold)
    G(pi) = sum_tau [ risk + ambiguity - info gain ]
Policy posterior q(pi) ∝ exp(-gamma * G(pi) + w_prior * ln E(pi)), with ln E(pi) = the VAE's log-likelihood of the
macro-actions (policy prior / habits). The winner's H-JEPA (F(L)) imagined endpoint is the planner goal;
commitment as in v3. Weights w_risk / w_amb / w_ig allow ablations (w_ig=1, others 0 = pure epistemic).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .action_vae import ActionVAE
from .goal_engine_gen import GoalEngineGen
from .goal_engine_v3 import V3Config
from .latent_dynamics import GaussEnsemble
from .patterns import PatternStream


@dataclass
class EFEWeights:
    w_ig: float = 1.0
    w_amb: float = 1.0
    w_risk: float = 1.0
    w_prior: float = 1.0
    gamma: float = 1.0
    sample: bool = False          # sample pi ~ q(pi) instead of argmax


class GoalEngineEFE(GoalEngineGen):
    def __init__(self, d: int, stream: PatternStream | None, vae: ActionVAE, dyn: GaussEnsemble,
                 cfg: V3Config = V3Config(), w: EFEWeights = EFEWeights(), seed: int = 0):
        super().__init__(d, stream, vae, cfg, seed=seed)
        self.dyn, self.w = dyn, w
        self.n_trans = 0

    def learn_transition(self, z, a, zn, fit_steps: int = 20):
        self.dyn.add(z, a, zn)
        self.n_trans += 1
        return self.dyn.fit(steps=fit_steps, device=z.device)

    def efe(self, z, acts):
        ig, amb, m, v = self.dyn.rollout_terms(z, acts)
        risk = 0.5 * (v + m ** 2 - 1 - v.log()).sum(-1)                            # KL(N(m,v) || N(0,1)), per step
        G = (self.w.w_risk * risk + self.w.w_amb * amb - self.w.w_ig * ig).sum(1)
        lnE = self.vae.log_prior(z, acts)
        logq = -self.w.gamma * G + self.w.w_prior * lnE
        return logq, dict(G=G, ig=ig.sum(1), amb=amb.sum(1), risk=risk.sum(1), lnE=lnE)

    def select(self, z, imagine_fn):
        self.z_ctx = z.reshape(1, -1)
        acts, _ = self.propose()
        logq, t = self.efe(z, acts)
        if self.w.sample:
            k = int(torch.distributions.Categorical(logits=logq).sample())
        else:
            k = int(torch.argmax(logq))
        goal = imagine_fn(z, acts[k:k + 1])[0]
        d0 = float((goal - z).norm())
        self.goal = dict(z0=z.detach(), goal=goal.detach(), act=acts[k].detach(), r=torch.full((1,), float('nan')),
                         d0=d0, best=d0, stall=0, age=0, mem_len=len(self.mem))
        self.stats['selections'] += 1
        return goal, {key: float(val[k]) for key, val in t.items()} | dict(k=k, spread_ig=float(t['ig'].std()))

    def finish_goal(self, z_reached):
        self.goal = None                       # learning happens per exact transition (learn_transition)
