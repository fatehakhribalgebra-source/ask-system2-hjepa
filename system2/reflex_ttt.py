"""Step 2 (DESIGN_ANIMAL L0): test-time adaptation of the reflex, H-JEPA levels 1 and 2.

AdaJEPA-style fast loop. Before every replan, take one gradient step on the latent prediction loss of the most recent
real transitions:

    L = sum_l  mean_i || f_l(z_l(t_i), u_l(t_i)) - z_l(t_i + 1) ||^2       (targets from the frozen encoders)

Only the FIRST transformer block of each adapted level's predictor moves (AdaJEPA's best split; JEPA-TTT freezes the
encoders so the RGM's places stay put). Everything is encoded once with no grad, so a step costs one predictor
forward/backward on a few transitions.

Geometry of the release AntMaze model (seed 42 checkpoint):
    level 1: one step = 5 env steps, action = 5 normalised env actions (40-d), encoder = pixels + proprio
    level 2: stride 2 over level 1 (10 env steps), action = SequenceEncoder over 2 level-1 action embeddings
The policy replans every 10 env steps, so each replan adds 2 level-1 and 1 level-2 transitions.

Persistence (JEPA-TTT): `state_dict()` / `load()` of the adapted blocks and the optimiser, so a later task can start
from what an earlier one learned. OpTTA's optimism term is not here yet.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

ENV_PER_L1 = 5          # env steps per level-1 step (action_block)


class ReflexTTT:
    def __init__(self, solver, levels=(1, 2), lr=1e-5, window=10, steps=1, optim='adam', device='cuda'):
        self.solver, self.levels, self.window, self.steps = solver, tuple(levels), int(window), int(steps)
        self.models = solver.level_models
        self.dev = device
        self.params = []
        for lv in self.levels:
            blk = self.models[lv - 1].predictor.predictor.transformer.layers[0]
            self.params += list(blk.parameters())
        for p in self.params:
            p.requires_grad_(True)
        self.frozen = [p.detach().clone() for p in self.params]     # the release weights, for the held-out check
        self.opt = (torch.optim.Adam(self.params, lr=lr) if optim == 'adam'
                    else torch.optim.SGD(self.params, lr=lr))
        self.z1, self.a1 = [], []           # level-1 embeddings at every 5th env step; level-1 action embeddings
        self._acts = []                     # normalised env actions since the last level-1 frame
        self.log = []                       # one row per update: env step, pre-update loss per level, post loss

    # ---------------------------------------------------------------- recording (called by the policy hooks)
    def record_frame(self, step: int, prepared: dict):
        """prepared = WorldModelPolicy._prepare_info output; keep a level-1 latent every ENV_PER_L1 env steps."""
        if step % ENV_PER_L1:
            return
        if self.z1 and len(self._acts) == ENV_PER_L1:          # close the previous level-1 action
            a = torch.as_tensor(np.concatenate(self._acts, -1), dtype=torch.float32, device=self.dev).reshape(1, 1, -1)
            with torch.no_grad():
                self.a1.append(self.models[0].action_embed(a)[0, 0])
        elif self.z1:                                           # a missing action (should not happen): restart
            self.z1, self.a1 = [], []
        self._acts = []
        with torch.no_grad():                                   # the planner's own encoding path (shapes, proprio)
            z = self.solver._encode_all_levels(prepared, key='pixels')['embed_1'][0, -1]
        self.z1.append(z)

    def record_action(self, norm_action):
        self._acts.append(np.asarray(norm_action, dtype=np.float32).reshape(-1))

    # ---------------------------------------------------------------- adaptation
    def _batches(self):
        """(level, z_t, u_t, z_t+1) for the last `window` level-1 transitions and the level-2 ones inside them."""
        n = len(self.a1)                                        # complete level-1 transitions
        if n < 1:
            return []
        lo = max(0, n - self.window)
        z1 = torch.stack(self.z1[lo:n + 1])                     # (T+1, D1)
        a1 = torch.stack(self.a1[lo:n])                         # (T, A1)
        out = []
        if 1 in self.levels:
            out.append((1, z1[:-1], a1, z1[1:]))
        if 2 in self.levels and z1.shape[0] >= 3:
            j2 = self.models[1]
            with torch.no_grad():
                enc = j2.encode({'embed_1': z1[None], 'action': torch.cat([a1, a1[-1:]], 0)[None]},
                                key='embed_1', chunk_temporal_inputs=True)
            z2, u2 = enc['embed_0'][0], enc['action_0'][0]          # (T2, D2), (T2, A2)
            if z2.shape[0] >= 2:
                out.append((2, z2[:-1], u2[:-1], z2[1:]))
        return out

    def _loss(self, batches, grad=True):
        losses = {}
        total = 0.0
        for lv, z, u, zn in batches:
            jepa = self.models[lv - 1]
            with torch.set_grad_enabled(grad):
                pred = jepa.predict(z[:, None], u[:, None])[:, -1]       # history 1: one transition per row
                l = F.mse_loss(pred, zn)
            losses[lv] = float(l.detach())
            total = total + l
        return total, losses

    def update(self, step: int):
        batches = self._batches()
        if not batches:
            return None
        row = {'step': int(step)}
        # Held-out check first: the newest transitions (2 at level 1, 1 at level 2) have not been trained on yet.
        # Their error under the adapted weights vs the release weights is the "done when" measure of L0.
        fresh = [(lv, z[-n:], u[-n:], zn[-n:]) for (lv, z, u, zn), n in
                 ((b, 2 if b[0] == 1 else 1) for b in batches)]
        _, row['fresh'] = self._loss(fresh, grad=False)
        with torch.no_grad():
            live = [p.detach().clone() for p in self.params]
            for p, q in zip(self.params, self.frozen):
                p.copy_(q)
            _, row['fresh_frozen'] = self._loss(fresh, grad=False)
            for p, q in zip(self.params, live):
                p.copy_(q)
        for k in range(self.steps):
            self.opt.zero_grad(set_to_none=True)
            loss, pre = self._loss(batches)
            if k == 0:
                row['pre'] = pre
            loss.backward()
            self.opt.step()
        self.opt.zero_grad(set_to_none=True)                   # planning backprops through these params too
        _, post = self._loss(batches, grad=False)
        row['post'] = post
        self.log.append(row)
        return row

    # ---------------------------------------------------------------- wiring + persistence
    def attach(self, policy):
        """Hook a WorldModelPolicy: record every prepared frame and every executed (normalised) action, and adapt
        right before each solver call."""
        ttt = self
        orig_prep, orig_solver = policy._prepare_info, policy.solver.solve
        proc = policy.process.get('action') if getattr(policy, 'process', None) else None

        def _prep(info_dict):
            out = orig_prep(info_dict)
            ttt.record_frame(int(policy._steps_taken), out)
            return out
        policy._prepare_info = _prep

        def _solve(info_dict, steps_taken, eval_budget):
            if steps_taken > 0:
                ttt.update(int(steps_taken))
            return orig_solver(info_dict, steps_taken=steps_taken, eval_budget=eval_budget)
        policy.solver.solve = _solve

        if proc is not None:                                    # capture the normalised action the env receives
            orig_inv = proc.inverse_transform

            def _inv(x):
                ttt.record_action(x)
                return orig_inv(x)
            proc.inverse_transform = _inv
        else:
            orig_get = policy.get_action

            def _get(info_dict, **kw):
                act = orig_get(info_dict, **kw)
                ttt.record_action(act)
                return act
            policy.get_action = _get
        return self

    def state_dict(self):
        return {'params': [p.detach().cpu() for p in self.params], 'opt': self.opt.state_dict(),
                'levels': self.levels}

    def load(self, sd):
        with torch.no_grad():
            for p, q in zip(self.params, sd['params']):
                p.copy_(q.to(p.device))
        try:
            self.opt.load_state_dict(sd['opt'])
        except (ValueError, KeyError):
            pass

    def summary(self):
        if not self.log:
            return {}
        out = {'updates': len(self.log)}
        for lv in self.levels:
            pre = [r['pre'][lv] for r in self.log if lv in r.get('pre', {})]
            post = [r['post'][lv] for r in self.log if lv in r.get('post', {})]
            if pre:
                k = max(1, len(pre) // 4)
                out[f'l{lv}_pre_first_quarter'] = float(np.mean(pre[:k]))
                out[f'l{lv}_pre_last_quarter'] = float(np.mean(pre[-k:]))
                out[f'l{lv}_post_mean'] = float(np.mean(post))
            fr = [r['fresh'][lv] for r in self.log if lv in r.get('fresh', {})]
            ff = [r['fresh_frozen'][lv] for r in self.log if lv in r.get('fresh_frozen', {})]
            if fr:
                out[f'l{lv}_fresh_adapted'] = float(np.mean(fr))    # held-out error, adapted weights
                out[f'l{lv}_fresh_frozen'] = float(np.mean(ff))     # same transitions, release weights
        return out
