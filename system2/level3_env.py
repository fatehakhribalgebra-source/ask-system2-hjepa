"""H-JEPA level 3 as an ENVIRONMENT for an external active-inference agent.

The agent observes level-3 latents z3 and acts with level-3 macro-actions a3 (4-d, the same action space level 3's
predictor F3 was trained with). One environment step = one level-3 step (20 real env steps on AntMaze).

  LatentLevel3Env : the environment is level 3's own predictor, z3' = F3(z3, a3)  (pure latent; for testing)
  RealLevel3Env   : a3 is executed in the real maze. Level 3 does NOT plan: its decision is fixed to the agent's a3
                    and the predicted next latent F3(z3, a3); H-JEPA levels 2 and 1 turn that into torques exactly as
                    they would turn a level-3 plan into torques. After 20 env steps z3' is re-encoded from the real
                    observation. The env also reports true xy / cells (never shown to the agent) for evaluation.
"""
from __future__ import annotations

import numpy as np
import torch

from .hjepa_bridge import ExplorationSolver


class LatentActionSolver(ExplorationSolver):
    """HierarchicalSolver whose top level is replaced by a fixed decision (a3, z3') supplied from outside."""

    fixed_action: torch.Tensor | None = None     # (1, 1, A)
    fixed_next: torch.Tensor | None = None       # (1, 1, D)

    @classmethod
    def from_solver(cls, s):
        new = super().from_solver(s)
        new.__class__ = cls
        new.fixed_action = new.fixed_next = None
        return new

    def solve(self, info_dict, steps_taken, eval_budget):
        if self.fixed_action is None:                 # commit-to-plan mode: full hierarchy toward goal_top
            return super().solve(info_dict, steps_taken, eval_budget)
        from hierarchical_solver import planning_call_index as _pci
        L = len(self.level_models)
        schedules = self._horizon_schedules(eval_budget=eval_budget)
        call_idx = _pci(replanning_interval=self._replanning_interval(), steps_taken=steps_taken)
        obs_emb = self._encode_all_levels(info_dict=info_dict, key='pixels')
        self.goal_top = self.fixed_next
        goal_emb = {f'embed_{L}': self.fixed_next}
        z_now = obs_emb[f'embed_{L}'][:, -1:]
        subgoal_latents = torch.cat([z_now, self.fixed_next], dim=1)        # (B, 2, D): [z3, F3(z3, a3)]
        subgoal_actions = self.fixed_action                                  # (B, 1, A)
        skipped_goal_level, result = None, None
        for level in range(L - 1, 0, -1):
            cfg = self._level_plan_config(level)
            horizon = schedules[level][call_idx]
            if level > 1 and horizon == 1:
                skipped_goal_level = L
                subgoal_latents = subgoal_actions = None
                continue
            li = dict(info_dict)
            li['embed_0'] = obs_emb[f'embed_{level}']
            li['action_cost_weight'] = cfg.action_cost_weight
            if subgoal_latents is None:
                gl = skipped_goal_level or L
                li['goal_embed_0'] = goal_emb[f'embed_{gl}']
                li['goal_embed_0_level'] = gl
                li['goal_embed_0_in_upper_space'] = gl > level
                li['dense_upper_goal_projection'] = gl > level
            else:
                k = self._level_plan_config(level + 1).num_subgoals if level + 1 < L else 1
                fp = 1 if subgoal_latents.shape[1] > 1 else 0
                fa = max(0, fp - 1)
                li['goal_embed_0'] = subgoal_latents[:, fp:fp + k].clone()
                li['goal_action'] = subgoal_actions[:, fa:fa + k].clone()
                li['goal_embed_0_in_upper_space'] = True
            result = self.level_solvers[level].solve(
                info_dict=li, planning_horizon=horizon, steps_taken=steps_taken, eval_budget=eval_budget)
            skipped_goal_level = None
            subgoal_latents = result['predictions']['predicted_embed_0']
            subgoal_actions = result['actions']
        return {'actions': result['actions']}


class LatentLevel3Env:
    """Pure-latent environment: z3' = F3(z3, a3)."""

    def __init__(self, top_jepa, z0: torch.Tensor):
        self.jepa, self.z0 = top_jepa, z0.reshape(-1)
        self.z = self.z0.clone()

    def reset(self, z0=None):
        self.z = (self.z0 if z0 is None else z0.reshape(-1)).clone()
        return self.z

    @torch.no_grad()
    def step(self, a3):
        emb = self.z.reshape(1, 1, -1)
        a = self.jepa.action_embed(a3.reshape(1, 1, -1))
        self.z = self.jepa.predict(emb, a)[:, -1].reshape(-1)
        return self.z, {}


class RealLevel3Env:
    """Real maze behind level 3. step(a3) runs `k_env` env steps with levels 2/1 realising a3."""

    def __init__(self, world, policy, solver, model, loco, cells, k_env: int = 20, recorder=None, tracker=None):
        self.world, self.policy, self.solver, self.model = world, policy, solver, model
        self.loco, self.cells, self.k, self.rec, self.tracker = loco, cells, k_env, recorder, tracker
        self.L = model.num_levels
        self.top = solver.level_models[self.L - 1]
        self.t = 0

    @torch.no_grad()
    def encode(self):
        info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.world.infos.items()}
        return self.solver._encode_all_levels(self.policy._prepare_info(info), key='pixels')[f'embed_{self.L}'].reshape(-1)

    def _track(self):
        infos = self.world.infos
        xy = np.asarray(infos['xy'][0]).reshape(-1)[-2:].copy()
        q = np.asarray(infos['qpos']).reshape(-1)[-15:]
        self.visited.add(tuple(np.asarray(self.loco.env.xy_to_ij(xy)).tolist()))
        if self.prev is not None:
            self.moving += float(np.linalg.norm(xy - self.prev) > 0.05)
        self.prev = xy
        up = float(q[3] ** 2 + q[6] ** 2 - q[4] ** 2 - q[5] ** 2)
        self.upright += up > 0.0
        if up <= 0.0 and self.first_flip is None:
            self.first_flip = self.t
        if self.t % 2 == 0:
            self.traj.append(xy.round(2).tolist())
        if self.t % 100 == 0:
            self.curve.append(len(self.visited & self.cells))
        return xy

    def reset(self, seed):
        self.world.reset(seed=seed)
        self.t, self.visited, self.moving, self.prev, self.upright, self.first_flip = 0, set(), 0, None, 0, None
        self.traj, self.curve = [], []
        if self.tracker is not None:
            self.tracker.reset()
        self._track()
        return self.encode()

    def step(self, a3, z3, label=''):
        """Execute macro-action a3 (A,) from latent z3 (D,): level 3's decision is fixed to (a3, F3(z3, a3))."""
        with torch.no_grad():
            nxt = self.top.predict(z3.reshape(1, 1, -1), self.top.action_embed(a3.reshape(1, 1, -1)))[:, -1:]
        self.solver.fixed_action = a3.reshape(1, 1, -1)
        self.solver.fixed_next = nxt
        self.policy.reset()
        transitions = []
        for _ in range(self.k):
            infos = self.world.infos
            infos['goal'] = infos['pixels']
            px0 = np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy()
            qp0, qv0 = np.asarray(infos['qpos']).reshape(-1)[-15:].copy(), np.asarray(infos['qvel']).reshape(-1)[-14:].copy()
            act = self.policy.get_action(infos)
            w = self.world
            w.states, w.rewards, w.terminateds, w.truncateds, w.infos = w.envs.step(act)
            self.t += 1
            xy = self._track()
            if self.tracker is not None:
                self.tracker.push(px0, qp0, qv0, act)
            if self.rec is not None:
                self.rec.frame(px0.astype(np.uint8), xy, None, f'{label} t={self.t} cells={len(self.visited & self.cells)}')
        z_next = self.encode()
        realised = self.tracker.transition() if self.tracker is not None else None   # exact (z3, a3_true, z3')
        return z_next, dict(predicted=nxt.reshape(-1), realised=realised)

    def step_plan(self, acts, z3, label=''):
        """Commit to a whole policy acts (H, A): the environment steers the full H-JEPA hierarchy toward the plan's
        predicted end state F3^H(z3, a_1..a_H) for H * k_env env steps. Returns the final z3 and the list of exact
        realised level-3 transitions (one per k_env steps) for the agent to learn from."""
        with torch.no_grad():
            zz = z3.reshape(1, 1, -1)
            for a3 in acts:
                zz = self.top.predict(zz, self.top.action_embed(a3.reshape(1, 1, -1)))[:, -1:]
        self.solver.fixed_action = self.solver.fixed_next = None
        self.solver.goal_top = zz
        self.policy.reset()
        realised = []
        for i in range(len(acts) * self.k):
            infos = self.world.infos
            infos['goal'] = infos['pixels']
            px0 = np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy()
            qp0, qv0 = np.asarray(infos['qpos']).reshape(-1)[-15:].copy(), np.asarray(infos['qvel']).reshape(-1)[-14:].copy()
            act = self.policy.get_action(infos)
            w = self.world
            w.states, w.rewards, w.terminateds, w.truncateds, w.infos = w.envs.step(act)
            self.t += 1
            xy = self._track()
            if self.tracker is not None:
                self.tracker.push(px0, qp0, qv0, act)
                if (i + 1) % self.k == 0:
                    tr = self.tracker.transition()
                    if tr is not None:
                        realised.append(tr)
            if self.rec is not None:
                self.rec.frame(px0.astype(np.uint8), xy, None, f'{label} t={self.t} cells={len(self.visited & self.cells)}')
        z_next = self.encode()
        return z_next, dict(predicted=zz.reshape(-1), realised=realised)

    def follow_road(self, waypoints, max_steps_per_wp: int = 100, reach_frac: float = 0.35, check: int = 20,
                    budget_left: int | None = None, label=''):
        """H-JEPA follows an imagined road: each waypoint (D,) becomes goal_top for the full hierarchy (its own
        goal-reaching planner) until the latent distance falls below reach_frac * initial distance or the per-waypoint
        step limit is hit. Returns final z3, exact realised transitions (every `check` steps) and per-waypoint log."""
        self.solver.fixed_action = self.solver.fixed_next = None
        realised, wp_log, steps = [], [], 0
        z = self.encode()
        for i, wp in enumerate(waypoints):
            goal_obs = None
            if isinstance(wp, dict):                  # remembered real observation: paper goal path (all levels)
                goal_obs, wp = wp, wp['z3']
                self.solver.full_goal = True
            else:
                self.solver.full_goal = False
                self.solver.goal_top = wp.reshape(1, 1, -1)
            self.policy.reset()
            d0 = float((wp - z).norm()); dist, n = d0, 0
            reached = False
            while n < max_steps_per_wp and (budget_left is None or steps < budget_left):
                infos = self.world.infos
                px0 = np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy()
                if goal_obs is not None:
                    infos['goal'] = goal_obs['pixels'].reshape(np.asarray(infos['pixels']).shape)
                    infos['goal_proprio'] = goal_obs['proprio'].reshape(*np.asarray(infos['pixels']).shape[:2], -1)
                else:
                    infos['goal'] = infos['pixels']
                qp0, qv0 = np.asarray(infos['qpos']).reshape(-1)[-15:].copy(), np.asarray(infos['qvel']).reshape(-1)[-14:].copy()
                act = self.policy.get_action(infos)
                w = self.world
                w.states, w.rewards, w.terminateds, w.truncateds, w.infos = w.envs.step(act)
                self.t += 1; n += 1; steps += 1
                xy = self._track()
                if self.tracker is not None:
                    self.tracker.push(px0, qp0, qv0, act)
                if self.rec is not None:
                    self.rec.frame(px0.astype(np.uint8), xy, None,
                                   f'{label} t={self.t} wp {i + 1}/{len(waypoints)} cells={len(self.visited & self.cells)}')
                if n % check == 0:
                    if self.tracker is not None:
                        tr = self.tracker.transition()
                        if tr is not None:
                            realised.append(tr)
                    z = self.encode(); dist = float((wp - z).norm())
                    if dist < reach_frac * d0:
                        reached = True
                        break
            wp_log.append(dict(wp=i, d0=d0, d_end=dist, steps=n, reached=reached))
            if budget_left is not None and steps >= budget_left:
                break
        self.solver.full_goal = False
        return self.encode(), dict(realised=realised, waypoints=wp_log, steps=steps)

    def stats(self, steps):
        return dict(coverage=len(self.visited & self.cells) / len(self.cells), cells=len(self.visited & self.cells),
                    total_cells=len(self.cells), curve=self.curve, motion=self.moving / max(1, steps - 1),
                    upright=self.upright / max(1, steps), first_flip=self.first_flip, traj=self.traj)
