"""Tschantz et al. (2020) "Reinforcement learning through active inference", official code (third_party/rl-inference,
MIT), adapted as H-JEPA's LEVEL-3 planner for goal reaching.

Kept exactly: pmbrl.models.EnsembleModel (probabilistic ensemble of state deltas + its loss), pmbrl.training.Normalizer,
pmbrl.control.Planner (CEM over continuous action sequences, sampled ensemble rollouts, Gaussian refit to the top
candidates), pmbrl.control.measures.InformationGain (kNN entropy of the ensemble's predicted deltas minus the average
Gaussian entropy). Return of a plan = preference + expl_scale * information gain (= -EFE), as in the paper.
Two edits:
  1. candidate actions are mapped through H-JEPA level 3's action LayerNorm before the ensemble, a = LN(u), so the
     agent only ever tries valid level-3 macro-actions (CEM still searches u, unconstrained);
  2. the learned reward model is replaced by the goal preference ln T(o) = -0.5 ||s - y_goal||^2 / sigma^2 with
     s, y_goal = whitened level-3 latents (the goal image encoded at level 3).
State = whitened z3 (32-d PCA view of the state VAE).
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

_REPO = pathlib.Path(__file__).resolve().parents[1] / 'third_party' / 'rl-inference'
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if 'pmbrl' not in sys.modules:                 # bare package: skip pmbrl/__init__ (it imports gym envs we don't use)
    import types
    _pkg = types.ModuleType('pmbrl'); _pkg.__path__ = [str(_REPO / 'pmbrl')]
    sys.modules['pmbrl'] = _pkg
from pmbrl.control.planner import Planner  # noqa: E402  (official)
from pmbrl.models.models import EnsembleModel  # noqa: E402  (official)
from pmbrl.training.normalizer import Normalizer  # noqa: E402  (official)

# official config for the larger continuous-control tasks (pmbrl/configs.py)
CFG = dict(ensemble_size=15, hidden_size=400, optimisation_iters=7, n_candidates=700, top_candidates=70,
           expl_scale=0.1, learning_rate=1e-3, epsilon=1e-8, grad_clip_norm=1000, batch_size=50)


class GoalPreference(nn.Module):
    """Replaces RewardModel: reward(s, a) = ln T(o) = -0.5 ||s - y_goal||^2 / sigma^2 (same call signature)."""

    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma, self.goal = sigma, None

    def forward(self, states, actions):
        return -0.5 * ((states - self.goal.reshape(1, -1)) ** 2).sum(-1) / self.sigma ** 2


class Level3Planner(Planner):
    """Official Planner; edits: actions -> LN(u) inside the rollout, and forward() returns the whole mean plan."""

    def __init__(self, *args, action_map=None, **kw):
        super().__init__(*args, **kw)
        self.action_map = action_map

    def perform_rollout(self, current_state, actions):
        return super().perform_rollout(current_state, self.action_map(actions))

    def plan(self, state: np.ndarray, horizon: int) -> torch.Tensor:
        self.plan_horizon = horizon
        state_t = torch.from_numpy(state).float().to(self.device)
        S = state_t.size(0)
        mean = torch.zeros(horizon, 1, self.action_size, device=self.device)
        std = torch.ones(horizon, 1, self.action_size, device=self.device)
        for _ in range(self.optimisation_iters):                                   # = official forward()
            actions = mean + std * torch.randn(horizon, self.n_candidates, self.action_size, device=self.device)
            states, delta_vars, delta_means = self.perform_rollout(state_t, actions)
            returns = torch.zeros(self.n_candidates, device=self.device)
            if self.use_exploration:
                returns += self.measure(delta_means, delta_vars) * self.expl_scale
            if self.use_reward:
                r = self.reward_model(states.view(-1, S), None).view(horizon, self.ensemble_size, self.n_candidates)
                returns += r.mean(1).sum(0) * self.reward_scale
            mean, std = self._fit_gaussian(actions, returns)
        return self.action_map(mean[:, 0])                                          # (H, A) valid macro-actions


def build(state_size=32, action_size=4, device='cuda', ensemble_size=CFG['ensemble_size'], hidden=CFG['hidden_size']):
    norm = Normalizer()
    ens = EnsembleModel(state_size + action_size, state_size, hidden, ensemble_size, norm, device=device)
    return ens, norm


def make_planner(ens, action_map, sigma=1.0, expl_scale=CFG['expl_scale'], use_exploration=True, device='cuda'):
    pref = GoalPreference(sigma).to(device)
    pl = Level3Planner(ens, pref, action_size=4, ensemble_size=ens.ensemble_size, plan_horizon=5,
                       optimisation_iters=CFG['optimisation_iters'], n_candidates=CFG['n_candidates'],
                       top_candidates=CFG['top_candidates'], use_reward=True, use_exploration=use_exploration,
                       expl_scale=expl_scale, strategy='information', device=device, action_map=action_map)
    return pl, pref
