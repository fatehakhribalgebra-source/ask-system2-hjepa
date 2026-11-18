"""Bridge between System 2 (goal engine) and a pretrained H-JEPA (kevinghst/H-JEPA).

- ExplorationSolver: HierarchicalSolver whose top-level goal is a latent supplied by System 2
  (no goal image). Anchoring to "true goal" latents is only done at the top level, because
  a latent goal has no counterpart in lower-level spaces.
- TopLevelReach: batched gradient-descent reachability of K candidate latents with the
  top-level predictor F(L) (the "feasibility" term of the EFE score).
- build_policy(): load checkpoint + eval config, return (WorldModelPolicy, solver, cfg).
"""
from __future__ import annotations

import os
import pathlib
import sys

import torch
from omegaconf import OmegaConf

HJEPA_DIR = pathlib.Path(__file__).resolve().parents[1] / 'H-JEPA' / 'h_jepa'
if str(HJEPA_DIR) not in sys.path:
    sys.path.insert(0, str(HJEPA_DIR))
os.environ.setdefault('MUJOCO_GL', 'glfw' if os.name == 'nt' else 'egl')

from hierarchical_solver import HierarchicalSolver, build_hierarchical_solver  # noqa: E402


class ExplorationSolver(HierarchicalSolver):
    """Top-down hierarchical planning toward a System-2 latent goal at the top level."""

    goal_top: torch.Tensor | None = None       # (B, 1, D_top)

    @classmethod
    def from_solver(cls, s: HierarchicalSolver) -> 'ExplorationSolver':
        new = cls.__new__(cls)
        new.__dict__.update(s.__dict__)
        new.goal_top = None
        return new

    def _encode_all_levels(self, info_dict, key='pixels'):
        if key != 'goal':
            return super()._encode_all_levels(info_dict, key)
        assert self.goal_top is not None, 'set solver.goal_top before planning'
        L = len(self.level_models)
        return {f'embed_{L}': self.goal_top}

    def solve(self, info_dict, steps_taken, eval_budget):
        # Same algorithm as HierarchicalSolver.solve, with anchoring restricted to the top level.
        from hierarchical_solver import planning_call_index as _pci
        subgoal_latents = subgoal_actions = None
        skipped_goal_level = None
        L = len(self.level_models)
        schedules = self._horizon_schedules(eval_budget=eval_budget)
        call_idx = _pci(replanning_interval=self._replanning_interval(), steps_taken=steps_taken)
        obs_emb = self._encode_all_levels(info_dict=info_dict, key='pixels')
        goal_emb = self._encode_all_levels(info_dict=info_dict, key='goal')
        result = None
        for level in range(L, 0, -1):
            cfg = self._level_plan_config(level)
            horizon = schedules[level][call_idx]
            if level > 1 and horizon == 1:
                skipped_goal_level = (skipped_goal_level or level) \
                    if cfg.horizon_one_goal_cost_space == 'upper' else None
                if skipped_goal_level is None:
                    skipped_goal_level = L     # latent goal only exists at the top level
                subgoal_latents = subgoal_actions = None
                continue
            li = dict(info_dict)
            li['embed_0'] = obs_emb[f'embed_{level}']
            li['action_cost_weight'] = cfg.action_cost_weight
            if subgoal_latents is None:
                gl = skipped_goal_level or level
                if f'embed_{gl}' not in goal_emb:
                    gl = L
                li['goal_embed_0'] = goal_emb[f'embed_{gl}']
                li['goal_embed_0_level'] = gl
                li['goal_embed_0_in_upper_space'] = gl > level
                li['dense_upper_goal_projection'] = gl > level
            else:
                k = self._level_plan_config(level + 1).num_subgoals
                fp = 1 if subgoal_latents.shape[1] > 1 else 0
                fa = max(0, fp - 1)
                nxt = subgoal_latents[:, fp:fp + k].clone()
                if level + 1 == L and fp + k >= subgoal_latents.shape[1]:
                    nxt[:, -1] = goal_emb[f'embed_{L}'].squeeze(1)    # anchor at top only
                li['goal_embed_0'] = nxt
                li['goal_action'] = subgoal_actions[:, fa:fa + k].clone()
                li['goal_embed_0_in_upper_space'] = True
            result = self.level_solvers[level].solve(
                info_dict=li, planning_horizon=horizon, steps_taken=steps_taken, eval_budget=eval_budget)
            skipped_goal_level = None
            subgoal_latents = result['predictions']['predicted_embed_0']
            subgoal_actions = result['actions']
        return {'actions': result['actions']}


class TopLevelReach:
    """Feasibility energy: how close can F(L) get to each candidate within H macro-steps."""

    def __init__(self, jepa, horizon: int = 4, steps: int = 30, lr: float = 0.1, action_dim: int = 4):
        self.jepa, self.H, self.steps, self.lr, self.adim = jepa, horizon, steps, lr, action_dim

    @torch.enable_grad()
    def __call__(self, z: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        K, D = goals.shape
        dev = goals.device
        acts = torch.zeros(K, self.H, self.adim, device=dev, requires_grad=True)
        opt = torch.optim.Adam([acts], lr=self.lr)
        z0 = z.reshape(1, 1, D).expand(K, 1, D).detach()
        for p in self.jepa.parameters():
            p.requires_grad_(False)
        for _ in range(self.steps):
            final = self._roll(z0, acts)
            loss = ((final - goals.detach()) ** 2).mean(-1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            final = self._roll(z0, acts)
            return ((final - goals) ** 2).mean(-1)

    def _roll(self, z0, acts):
        emb = z0
        for t in range(self.H):
            a = self.jepa.action_embed(acts[:, t:t + 1])
            emb = self.jepa.predict(emb[:, -1:], a)[:, -1:]
        return emb[:, -1]


class TopLevelImagine:
    """Roll F(L) forward under macro-action sequences; returns the final imagined latent."""

    def __init__(self, jepa):
        self.jepa = jepa

    @torch.no_grad()
    def __call__(self, z: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        K, H, _ = acts.shape
        emb = z.reshape(1, 1, -1).expand(K, 1, -1)
        for t in range(H):
            a = self.jepa.action_embed(acts[:, t:t + 1])
            emb = self.jepa.predict(emb[:, -1:], a)[:, -1:]
        return emb[:, -1]


def build_policy(ckpt: str, eval_config: str = 'ant_l3', cache_dir: str | None = None,
                 overrides: dict | None = None):
    """Load an eval config + checkpoint, return (policy, solver, cfg, model)."""
    from planning_eval import (_build_eval_process_from_policy_normalizer, _drop_levels_above_model,
                               img_transform, load_model)
    from eval_config_utils import _build_policy_plan_config
    import stable_worldmodel as swm

    cfg = OmegaConf.load(HJEPA_DIR / 'config' / 'eval' / f'{eval_config}.yaml')
    cfg.policy = ckpt
    cfg.cache_dir = cache_dir or os.environ.get('HJEPA_HOME')
    for k, v in (overrides or {}).items():
        OmegaConf.update(cfg, k, v)
    model = load_model(cfg)
    _drop_levels_above_model(cfg, model.num_levels)
    solver = ExplorationSolver.from_solver(build_hierarchical_solver(cfg, model))
    process = _build_eval_process_from_policy_normalizer(cfg, model=model)
    transform = {'pixels': img_transform(cfg), 'goal': img_transform(cfg)}
    policy = swm.policy.WorldModelPolicy(solver=solver, config=_build_policy_plan_config(cfg),
                                         process=process, transform=transform,
                                         eval_budget=int(cfg.eval.eval_budget))
    return policy, solver, cfg, model
