"""Paper AntMaze goal-image eval with a swappable LEVEL-3 planner (everything else = the paper's pipeline).

--planner gd       H-JEPA's own gradient planner (baseline)
--planner fountas  official Fountas et al. deep active inference (adapted, system2/fountas_z3.py): MCTS on EFE over
                   12 real macro-action prototypes, goal = preference. Its chosen macro-actions are rolled out by the
                   ORIGINAL level-3 solver (_rollout_final_actions), so levels 2 and 1 receive the identical format.
One task per process (Windows/glfw). The driver mode (--tasks a b c) relaunches itself per task.

  python experiments/eval_l3_planner.py --planner fountas --tasks 0 1 2 3 4 5 6 7 8 9 10 11 12
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
HJ = ROOT / 'H-JEPA' / 'h_jepa'


class AIFLevel3:
    """Stands in for level_solvers[3]: same solve() signature and output format."""

    def __init__(self, orig, agent_fn, name, pass_steps=False):
        self.__dict__['_orig'] = orig
        self.__dict__['_agent_fn'] = agent_fn
        self.__dict__['_pass_steps'] = pass_steps
        self.__dict__['solve_records'] = []
        self.__dict__['_name'] = name
        self.__dict__['decisions'] = []

    def __getattr__(self, k):
        return getattr(self.__dict__['_orig'], k)

    def solve(self, info_dict, planning_horizon, steps_taken, eval_budget):
        import torch
        t0 = time.time()
        z = info_dict['embed_0'][:, -1]
        g = info_dict['goal_embed_0'].reshape(z.shape[0], -1, z.shape[-1])[:, -1]
        if self._pass_steps:
            acts, info = self._agent_fn(z[0], g[0], int(planning_horizon), steps_taken=int(steps_taken))
        else:
            acts, info = self._agent_fn(z[0], g[0], int(planning_horizon))
        if acts is None:                                    # route finished: H-JEPA's own planner does the rest
            self.decisions.append(dict(step=int(steps_taken), horizon=int(planning_horizon), **info))
            return self._orig.solve(info_dict=info_dict, planning_horizon=planning_horizon,
                                    steps_taken=steps_taken, eval_budget=eval_budget)
        acts = acts.reshape(1, planning_horizon, -1).to(z.device)
        preds = self._orig._rollout_final_actions(info_dict, acts)
        self.solve_records.append({'level': 3, 'horizon': int(planning_horizon), 'planner': self._name,
                                   'wall_clock': time.time() - t0})
        self.decisions.append(dict(step=int(steps_taken), horizon=int(planning_horizon), **info))
        return {'actions': acts.detach().cpu(), 'predictions': preds}


class PaperLevel:
    """Stands in for level_solvers[n] (n = observed level): the paper-style RGM observes z_n, plans by inductive
    inference and PREDICTS the next latent; it returns that prediction as the plan the level below must follow (the
    reflex). Predictions are [now, target, target] so H-JEPA's last-subgoal anchoring never replaces the target with
    the goal image. In a goal state (or without a plan) it hands over to H-JEPA's own planner for this level."""

    def __init__(self, orig, rgm, ex, level, outer, lr=0.0, obs_every=0):
        d = self.__dict__
        d['_orig'], d['_rgm'], d['_ex'], d['_level'], d['_outer'], d['_lr'] = orig, rgm, ex, level, outer, lr
        d['_outer_obs_every'] = obs_every
        d['decisions'], d['solve_records'], d['_target'] = [], [], None

    def __getattr__(self, k):
        return getattr(self.__dict__['_orig'], k)

    def solve(self, info_dict, planning_horizon, steps_taken, eval_budget):
        import numpy as np
        import torch
        lv = self._level
        z = info_dict['embed_0'][:, -1]                                     # (1, D) current latent at this level
        with torch.no_grad():
            zg = self._outer._encode_all_levels(info_dict, key='goal')[f'embed_{lv}'].reshape(1, -1)
        every = 10 * (lv - 1)                                               # env steps per latent step
        look = self._outer_obs_every or every                               # extra looks between latent steps
        tick = steps_taken % every == 0
        observe = tick or steps_taken % look == 0 or self._target is None
        if observe:
            zn = z[0].cpu().numpy()
            target, info = self._rgm.step(zn, zg[0].cpu().numpy(), observe=True, lr=self._lr,
                                          tick=tick or self._target is None)
            self.__dict__['_target'] = target
            lat = None
            if target is not None and getattr(self._rgm, 'live', False):  # states grow: look the exemplar up now
                lat = self._rgm.exemplar(target)
            elif target is not None and hasattr(self._rgm, 'ctx_pairs'):   # reachable exemplar from here
                lat = self._rgm.target_latent(zn, info['s'], target)
                info = info | dict(ctx=lat is not None)
            self.__dict__['_tgt_lat'] = lat
        else:
            target, info = self._target, dict(held=True)
        if target is None:
            self.decisions.append(dict(step=int(steps_taken), handover=True, **info))
            return self._orig.solve(info_dict=info_dict, planning_horizon=planning_horizon,
                                    steps_taken=steps_taken, eval_budget=eval_budget)
        lat = self.__dict__.get('_tgt_lat')
        tgt = (torch.as_tensor(lat) if lat is not None else self._ex[target]).reshape(1, -1).to(z.device, z.dtype)
        preds = torch.stack([z, tgt, tgt], dim=1)                           # (1, 3, D)
        acts = torch.zeros(1, 2, 4, device=z.device)
        self.decisions.append(dict(step=int(steps_taken), horizon=int(planning_horizon), **info))
        self.solve_records.append({'level': lv, 'horizon': int(planning_horizon), 'planner': 'rgm_paper'})
        return {'actions': acts.cpu(), 'predictions': {'predicted_embed_0': preds}}


def run_task(a, t):
    os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
    os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'
    sys.path.insert(0, str(HJ)); sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import planning_eval as pe
    import stable_worldmodel as swm

    cfg = OmegaConf.load(HJ / 'config' / 'eval' / f'{a.env}_l{a.levels}.yaml')
    cfg.seed = a.seed
    name = f'{a.env}_hjepa_l{a.levels}'
    cfg.policy = str(ROOT / 'data' / 'ckpts' / a.env / name / f'seed{a.seed}' / f'{name}_object.ckpt')
    cfg.cache_dir = os.environ['HJEPA_HOME']
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if a.planner == 'rgm_paper' and a.rgm_obs_level == 2:
        # the RGM hands level 1 a predicted observation, not a macro-action: the reflex follows the observation only
        OmegaConf.update(cfg, 'hierarchical_plan_config.level1.action_cost_weight', 0.0)
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    transform = {'pixels': pe.img_transform(cfg), 'goal': pe.img_transform(cfg)}
    process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
    policy = pe._build_policy(cfg, process, transform, model=None)
    solver = policy.solver
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values() if hasattr(s, 'torch_gen')]
    for gen, seed in gens:
        gen.manual_seed(seed + t)                           # identical to the paper's chunked eval

    dev = 'cuda'
    if a.planner not in ('gd', 'rgm_paper'):           # AntMaze level-3 assets for the action planners
        from system2.active_inference import StateVAE
        ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
        svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
        protos = torch.load(ROOT / 'data' / 'z3_game_assets.pt', map_location=dev, weights_only=False)['protos'].to(dev)
    if a.planner == 'fountas':
        from system2.fountas_z3 import ActiveInferenceModelZ3, MCTSParams, active_inference_mcts
        sd = torch.load(ROOT / 'data' / 'fountas_z3.pt', map_location=dev, weights_only=False)
        fm = ActiveInferenceModelZ3(s_dim=sd['s_dim'], o_dim=sd['o_dim'], pi_dim=sd['pi_dim'], sigma_goal=a.sigma_goal)
        fm.load_state_dict(sd); fm.train(False)
        params = MCTSParams(); params.repeats = a.mcts_repeats

        def agent_fn(z, g, H):
            with torch.no_grad():
                fm.goal = svae.zn(g.reshape(1, -1))[0]
                path, n = active_inference_mcts(fm, svae.zn(z.reshape(1, -1))[0], params)
            path = (path + [path[-1]] * H)[:H]
            return protos[path], dict(path=path, mcts_iters=n)
        solver.level_solvers[3] = AIFLevel3(solver.level_solvers[3], agent_fn, 'fountas')

    if a.planner == 'tschantz':
        from system2.tschantz_l3 import build, make_planner
        tk = torch.load(ROOT / 'data' / 'tschantz_l3.pt', map_location=dev, weights_only=False)
        ens, norm = build(tk['state_size'], tk['action_size'], dev)
        ens.load_state_dict(tk['state_dict']); norm.__dict__.update(tk['normalizer']); ens.eval()
        ln = list(solver.level_models[-1].action_encoder.children())[-1]
        amap = lambda u: ln(u)
        planner, pref = make_planner(ens, amap, sigma=a.sigma_goal, expl_scale=a.expl_scale, device=dev)

        E = ens.ensemble_size
        route = dict(states=None, actions=None, p=0, plays=0)

        @torch.no_grad()
        def imagine_step(s, a3):                             # the z3 game: the agent's ensemble (mean of members)
            dm, _ = ens(s.reshape(1, 1, -1).expand(E, 1, -1), a3.reshape(1, 1, -1).expand(E, 1, -1))
            return s + dm.mean(0)[0]

        @torch.no_grad()
        def play_to_goal(s0, goal):
            states, acts, s = [s0], [], s0
            for _ in range(a.max_route):
                a3 = planner.plan(s.cpu().numpy(), max(5, a.plan_len))[0]
                s = imagine_step(s, a3); acts.append(a3); states.append(s)
                if float((s - goal).norm()) < a.near:
                    break
            route.update(states=torch.stack(states), actions=torch.stack(acts), p=0, plays=route['plays'] + 1)

        def agent_fn(z, g, H):
            with torch.no_grad():
                pref.goal = svae.zn(g.reshape(1, -1))[0]
                s0 = svae.zn(z.reshape(1, -1))[0]
                d_goal = float((s0 - pref.goal).norm())
                if not a.route:
                    acts = planner.plan(s0.cpu().numpy(), max(H, a.plan_len))[:H]
                    return acts, dict(dist_to_goal_whitened=d_goal)
                if route['states'] is None:
                    play_to_goal(s0, pref.goal)
                p = route['p']
                win = route['states'][p:p + 6]
                p = p + int((win - s0).norm(dim=-1).argmin())
                if float((route['states'][p] - s0).norm()) > a.drift:          # drifted: play again from here
                    play_to_goal(s0, pref.goal); p = 0
                route['p'] = p
                info = dict(dist_to_goal_whitened=d_goal, route_len=len(route['actions']), route_pos=p,
                            plays=route['plays'],
                            route_end_to_goal=float((route['states'][-1] - pref.goal).norm()))
                if p >= len(route['actions']):                                 # route done: H-JEPA finishes
                    return None, info | dict(handover=True)
                acts = route['actions'][p:p + H]
                acts = torch.cat([acts, acts[-1:].expand(H - len(acts), -1)]) if len(acts) < H else acts
            return acts, info
        solver.level_solvers[3] = AIFLevel3(solver.level_solvers[3], agent_fn, 'tschantz')

    if a.planner == 'rgm':
        from system2.rgm_l3 import RGMLevel3
        rgm = RGMLevel3.from_state(torch.load(a.rgm_path, weights_only=False), levels=a.rgm_levels)
        P = torch.as_tensor(rgm.P, dtype=torch.float32, device=dev)
        route = dict(acts=None, states=None, t0=0, plans=0)

        def rollout_states(y, acts):                         # expected discrete states along a committed route
            q = rgm.posterior(y); out = [int(q.argmax())]
            for u in acts:
                q = rgm.b1[:, :, u] @ q; q = q / max(q.sum(), 1e-12); out.append(int(q.argmax()))
            return out

        def agent_fn(z, g, H, steps_taken=0):
            with torch.no_grad():
                y = svae.zn(z.reshape(1, -1))[0].cpu().numpy()
                yg = svae.zn(g.reshape(1, -1))[0].cpu().numpy()
            if a.rgm_mode == 'closed':
                acts, info = rgm.plan(y, yg, H)
                if acts is None:                             # in a goal state (or no path): H-JEPA finishes
                    return None, info | dict(handover=True)
                return P[acts], info
            if a.rgm_mode == 'bridge':
                # closed loop + learning from reality + RGM/F3 product of experts with precisions learned from reality
                q_now = rgm.posterior(y)
                top = solver.level_models[-1]

                @torch.no_grad()
                def f3_post(us):
                    zz = z.reshape(1, 1, -1).expand(len(us), 1, -1)
                    aa = top.action_embed(P[torch.as_tensor(us, device=P.device)].reshape(len(us), 1, -1))
                    zn = top.predict(zz, aa)[:, -1]
                    return [rgm.posterior(v) for v in svae.zn(zn).cpu().numpy()]

                st_ = route.setdefault('bridge', dict(acts=None, t_dec=0, win=None, n=0, s_rgm=0.0, s_f3=0.0,
                                                      log=[]))
                cal = getattr(rgm, 'calib', {}) or {}
                prior_n = 10.0

                def w_f3():
                    if a.bridge_f3 >= 0:
                        return a.bridge_f3
                    m_r = (cal.get('ll_rgm', -0.7) * prior_n + st_['s_rgm']) / (prior_n + st_['n'])
                    m_f = (cal.get('ll_f3', -0.7) * prior_n + st_['s_f3']) / (prior_n + st_['n'])
                    return float(np.clip(np.exp(m_f - m_r), 0.1, 3.0))

                info, force = {}, False
                on_grid = steps_taken % 20 == 0
                win = st_['win']
                if on_grid and win is not None and steps_taken - win['t'] == 20:      # feedback from reality
                    ll_r = float(np.log(win['pred_rgm'] @ q_now + 1e-12))
                    ll_f = float(np.log(win['pred_f3'] @ q_now + 1e-12))
                    st_['n'] += 1; st_['s_rgm'] += ll_r; st_['s_f3'] += ll_f
                    if a.rgm_learn:
                        rgm.learn(win['q'], win['u'], q_now, lr=a.rgm_lr)
                    info |= dict(surprise=-ll_r, ll_f3=ll_f)
                    force = -ll_r > rgm.surprise_thresh
                if rgm.goal_states(yg)[int(q_now.argmax())]:
                    return None, dict(reason='at_goal', handover=True)
                if st_['acts'] is None or force or (steps_taken - st_['t_dec']) >= a.bridge_hold * 10:
                    acts, preds, dinfo = rgm.bridge_decide(q_now, yg, f3_post, w_f3(), H=max(H, 5))
                    if acts is None:                                                 # no progress move: induction
                        acts, pinfo = rgm.plan(y, yg, max(H, 5))
                        if acts is None:
                            return None, pinfo | dict(handover=True)
                        dinfo = pinfo | dict(fallback=True)
                    st_.update(acts=[int(u) for u in acts], t_dec=steps_taken)
                    info |= dinfo | dict(decided=True, forced=bool(force))
                k = (steps_taken - st_['t_dec']) // 20
                seq = st_['acts'][k:k + H] or st_['acts'][-1:]
                seq = seq + [seq[-1]] * (H - len(seq))
                if on_grid:                                                          # open a feedback window
                    u0 = int(seq[0])
                    pr = rgm.b1[:, :, u0] @ q_now
                    st_['win'] = dict(t=steps_taken, q=q_now, u=u0, pred_rgm=pr / max(pr.sum(), 1e-12),
                                      pred_f3=f3_post([u0])[0])
                return P[np.asarray(seq)], info | dict(s=int(q_now.argmax()), w=round(w_f3(), 3), k=int(k))
            if a.rgm_mode == 'feedback':
                # two realities: inner loop checks the route in F3 before committing; outer loop compares every real
                # level-3 step (20 env steps) with the committed, F3-checked expectation and replans on surprise
                q_now = rgm.posterior(y)
                if rgm.goal_states(yg)[int(q_now.argmax())]:
                    return None, dict(reason='at_goal', handover=True)
                top = solver.level_models[-1]

                @torch.no_grad()
                def imagine(acts_):
                    zz, out = z.reshape(1, 1, -1), []
                    for u in acts_:
                        zz = top.predict(zz, top.action_embed(P[int(u)].reshape(1, 1, -1)))[:, -1:]
                        out.append(svae.zn(zz.reshape(1, -1))[0].cpu().numpy())
                    return out

                info, why = {}, None
                if route['acts'] is None:
                    why = 'start'
                else:
                    p = (steps_taken - route['t0']) // 20
                    if p > route['checked']:                    # a real level-3 step finished: feedback
                        e = route['exp'][min(p, len(route['exp'])) - 1]
                        s_val = rgm.surprise(e, q_now)
                        route['surprises'].append(round(s_val, 3))
                        if a.rgm_learn and (p - 1) in route['q_at'] and p - 1 < len(route['acts']):
                            # continual learning from a single real level-3 transition (skipped when level 3 was not
                            # called for a while and several actions passed between two observations)
                            rgm.learn(route['q_at'][p - 1], route['acts'][p - 1], q_now)
                        route['q_at'][p] = q_now
                        route['checked'] = p
                        info['surprise'] = s_val
                        if a.rgm_every_action:                  # replan from the real state after every action
                            why = 'every_action'
                        elif s_val > rgm.surprise_thresh:
                            why = 'surprise'
                    if why is None and p >= len(route['acts']):
                        why = 'route_end'
                if why is not None:
                    acts, exp, pinfo = rgm.verified_route(y, yg, a.max_route, imagine)
                    if acts is None:
                        return None, pinfo | dict(handover=True, why=why)
                    route.update(acts=[int(u) for u in acts], exp=exp, t0=steps_taken, checked=0,
                                 q_at={0: q_now}, plans=route['plans'] + 1, surprises=route.get('surprises', []))
                    info |= dict(why=why, attempts=pinfo['attempts'], blocked=pinfo['blocked'],
                                 inner=pinfo['inner'])
                p = (steps_taken - route['t0']) // 20
                seq = route['acts'][p:p + H] or route['acts'][-1:]
                seq = seq + [seq[-1]] * (H - len(seq))
                return P[np.asarray(seq)], info | dict(route_pos=int(p), route_len=len(route['acts']),
                                                       plans=route['plans'], s=int(q_now.argmax()))
            if a.rgm_mode == 'adaptive':
                # EFE choice at every call: continue the committed route or switch to a fresh plan
                p = 0 if route['acts'] is None else (steps_taken - route['t0']) // 20
                remaining = [] if route['acts'] is None else route['acts'][p:]
                replan, fresh, info, terms = rgm.commit_or_replan(y, yg, remaining, a.max_route, a.switch_cost)
                if fresh is None and info.get('reason') == 'at_goal':
                    return None, info | dict(handover=True)
                if replan and fresh is not None:
                    route.update(acts=[int(u) for u in fresh], t0=steps_taken, plans=route['plans'] + 1)
                    p, remaining = 0, route['acts']
                if not remaining:                            # nothing to continue and no plan: H-JEPA finishes
                    return None, info | dict(handover=True)
                seq = list(remaining[:H]); seq = seq + [seq[-1]] * (H - len(seq))
                return P[np.asarray(seq)], {k: v for k, v in info.items() if k != 'reason'} | terms | dict(
                    replanned=bool(replan), route_pos=int(p), route_len=len(route['acts']), plans=route['plans'])
            # commit: one macro-action per level-3 step (20 env steps), replan only on drift / end of route
            s_now = int(rgm.posterior(y).argmax())
            p = (steps_taken - route['t0']) // 20
            drift = route['acts'] is None or p >= len(route['acts']) or \
                s_now not in route['states'][max(0, p - 2): p + 3]
            info = {}
            if drift:
                acts, info = rgm.plan(y, yg, a.max_route)
                if acts is None:
                    return None, info | dict(handover=True)
                k = max(1, int(info.get('plan_len', len(acts))))               # real route, without padding
                route.update(acts=list(acts[:k]), states=rollout_states(y, acts[:k]), t0=steps_taken,
                             plans=route['plans'] + 1)
                p = 0
            seq = route['acts'][p:p + H]
            seq = seq + [seq[-1]] * (H - len(seq))
            return P[np.asarray(seq)], info | dict(route_pos=int(p), route_len=len(route['acts']),
                                                    plans=route['plans'], s=s_now)
        solver.level_solvers[3] = AIFLevel3(solver.level_solvers[3], agent_fn, 'rgm', pass_steps=True)

    if a.planner == 'rgm_paper':
        import pickle
        from system2.rgm_paper import PaperRGM
        path = a.rgm_paper_path or str(ROOT / 'data' / f'rgm_paper_z{a.rgm_obs_level}_k64.pkl')
        state = None
        if not a.rgm_live:
            with open(path, 'rb') as fh:
                state = pickle.load(fh)
        if a.rgm_live:                                      # no offline map: the RGM starts empty and grows
            from system2.rgm_live import LiveRGM
            persist = pathlib.Path(a.rgm_persist) if a.rgm_persist else None
            if persist is not None and persist.exists():
                with open(persist, 'rb') as fh:
                    prgm = pickle.load(fh)
                prgm.reset()
            else:
                prgm = LiveRGM()
            top = solver.level_models[len(solver.level_models) - 1]
            ln = list(top.action_encoder.children())[-1]    # level 3's action manifold (LayerNorm)
            g = torch.Generator().manual_seed(a.seed)
            with torch.no_grad():
                live_protos = ln(torch.randn(12, ln.normalized_shape[0], generator=g).to(dev))

            def f3_latents(zn, _top=top, _P=live_protos):
                with torch.no_grad():
                    zz = torch.as_tensor(zn, dtype=torch.float32, device=dev).reshape(1, 1, -1).repeat(len(_P), 1, 1)
                    return _top.predict(zz, _top.action_embed(_P[:, None]))[:, -1].float().cpu().numpy()
            prgm.f3_latents = f3_latents
            if a.rgm_dream:                                 # offline structure learning inside F3 dreams
                gb = torch.Generator(device=dev).manual_seed(a.seed * 1000 + t)

                def f3_batch(Zb, _top=top, _ln=ln, _g=gb):
                    with torch.no_grad():
                        u = torch.randn(len(Zb), _ln.normalized_shape[0], generator=_g, device=dev)
                        zz = torch.as_tensor(Zb, dtype=torch.float32, device=dev)[:, None]
                        return _top.predict(zz, _top.action_embed(_ln(u)[:, None]))[:, -1].float().cpu().numpy()
                prgm.dreaming, prgm.f3_batch = True, f3_batch
        elif a.rgm_agent:                                     # RGM agent: + control evidence, surprise, EFE, F3 live
            from system2.rgm_agent import RGMAgent
            persist = pathlib.Path(a.rgm_persist) if a.rgm_persist else None
            if persist is not None and persist.exists():    # continual protocol: model carried from earlier tasks
                with open(persist, 'rb') as fh:
                    prgm = pickle.load(fh)
                prgm.reset()
            else:
                prgm = RGMAgent.from_paper(state)
                sq = np.load(a.rgm_calib_seqs)
                prgm.calibrate(sq[f'z{a.rgm_obs_level}'])
                if a.rgm_f3:
                    prgm.add_transitions(np.load(a.rgm_f3)['N_imag'], a.rgm_f3_weight)
            if a.rgm_f3:                                    # live F3: states reachable in one level-3 step from now
                top = solver.level_models[len(solver.level_models) - 1]
                protos = torch.as_tensor(np.load(a.rgm_f3)['protos'], dtype=torch.float32, device=dev)

                def f3_states(zn, _m=prgm, _top=top, _P=protos):
                    with torch.no_grad():
                        zz = torch.as_tensor(zn, dtype=torch.float32, device=dev).reshape(1, 1, -1).repeat(len(_P), 1, 1)
                        nxt = _top.predict(zz, _top.action_embed(_P[:, None]))[:, -1].float().cpu().numpy()
                    Y = _m.white(nxt)
                    return set(((Y[:, None] - _m.C[None]) ** 2).sum(-1).argmin(1).tolist())
                prgm.f3 = f3_states
        elif a.rgm_full:                                    # fuller RGM: filtering, gated learning, F3 transitions
            from system2.rgm_full import FullRGM
            prgm = FullRGM.from_paper(state)
            prgm.filtering, prgm.gated = not a.rgm_no_filter, not a.rgm_no_gate
            if a.rgm_f3:
                prgm.add_transitions(np.load(a.rgm_f3)['N_imag'], a.rgm_f3_weight)
        else:
            prgm = PaperRGM.from_state(state)
        prgm.stuck_k, prgm.relax, prgm.lookahead = a.rgm_stuck, a.rgm_relax, a.rgm_lookahead
        if a.rgm_ctx_seqs:
            sq = np.load(a.rgm_ctx_seqs)
            prgm.set_context(sq[f'z{a.rgm_obs_level}'], sq[f'ep{a.rgm_obs_level}'])
        ex = None if a.rgm_live else torch.as_tensor(prgm.ex, dtype=torch.float32, device=dev)
        lv = a.rgm_obs_level
        solver.level_solvers[lv] = PaperLevel(solver.level_solvers[lv], prgm, ex, lv, solver,
                                              lr=a.rgm_lr if a.rgm_learn else 0.0, obs_every=a.rgm_observe_every)

    payload = torch.load(str(HJ / cfg.load_eval_trajs_path), map_location='cpu', weights_only=False)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    chunk = {**payload, 'data': [dict(payload['data'][t])], 'episodes_idx': [t]}
    img = int(cfg.eval.img_size)
    world = swm.World(**{**world_cfg, 'num_envs': 1}, image_shape=(img, img))
    world.set_policy(policy)
    t0 = time.time()
    try:
        with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
            m = world.evaluate_from_dataset(
                None, episodes_idx=None, start_steps=None, goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget, callables=callables, load_eval_trajs_path=chunk, process=process,
                start_state_mode=cfg.eval.start_state_mode, goal_state_mode=cfg.eval.goal_state_mode)
    finally:
        world.close()
    l3 = solver.level_solvers[a.rgm_obs_level if a.planner == 'rgm_paper' else 3]
    r = dict(task=t, planner=a.planner, success=bool(m['episode_successes'][0]),
             steps_to_success=float(m['steps_to_success'][0]), seconds=time.time() - t0,
             decisions=getattr(l3, 'decisions', None))
    if a.planner == 'rgm_paper':
        r['rgm_stats'] = {k: int(v) for k, v in prgm.stats.items()}
        if (a.rgm_agent or a.rgm_live) and a.rgm_persist:
            with open(a.rgm_persist, 'wb') as fh:
                pickle.dump(prgm, fh)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planner', default='gd', choices=['gd', 'fountas', 'tschantz', 'rgm', 'rgm_paper'])
    ap.add_argument('--rgm-obs-level', type=int, default=3, choices=[2, 3],
                    help='rgm_paper: H-JEPA level the RGM observes (its reflex is the level below)')
    ap.add_argument('--rgm-paper-path', default='', help='rgm_paper: model pickle (default rgm_paper_z<level>_k64.pkl)')
    ap.add_argument('--rgm-stuck', type=int, default=0,
                    help='rgm_paper: block a move after this many observations without reaching its target (0 = off)')
    ap.add_argument('--rgm-relax', action='store_true',
                    help='rgm_paper: on no path, retry with all counted edges / no blocks / wider goal before handover')
    ap.add_argument('--rgm-ctx-seqs', default='',
                    help='rgm_paper: latent sequences npz; hand down the real successor latent reached in the data '
                         'from the place nearest to now, instead of the state medoid')
    ap.add_argument('--rgm-lookahead', type=int, default=1,
                    help='rgm_paper: hand down the k-th next distinct state of the plan (1 = next state, as before)')
    ap.add_argument('--rgm-observe-every', type=int, default=0,
                    help='rgm_paper: also look at the state every this many env steps (0 = once per latent step); '
                         'the target advances as soon as a hop is reached, temporal depth still ticks per latent step')
    ap.add_argument('--rgm-full', action='store_true',
                    help='rgm_paper: use system2/rgm_full.FullRGM (filtering + gated continual learning)')
    ap.add_argument('--rgm-agent', action='store_true',
                    help='rgm_paper: system2/rgm_agent.RGMAgent (full + control evidence, surprise, EFE fallback, live F3)')
    ap.add_argument('--rgm-calib-seqs', default='', help='rgm-agent: latent sequences for the surprise threshold')
    ap.add_argument('--rgm-live', action='store_true',
                    help='rgm_paper: system2/rgm_live.LiveRGM -- no offline map, states and moves learned while playing')
    ap.add_argument('--rgm-dream', action='store_true', help='rgm-live: structure learning on F3 dreams')
    ap.add_argument('--rgm-persist', default='',
                    help='rgm-agent: carry the learned model from task to task through this pickle (continual protocol)')
    ap.add_argument('--rgm-no-filter', action='store_true', help='rgm-full: decode each latent alone')
    ap.add_argument('--rgm-no-gate', action='store_true', help='rgm-full: plain (ungated) continual learning')
    ap.add_argument('--rgm-f3', default='', help='rgm-full: imagined F3 transitions npz (build_f3_transitions.py)')
    ap.add_argument('--rgm-f3-weight', type=float, default=0.1, help='rgm-full: weight of one imagined transition')
    ap.add_argument('--rgm-path', default=str(ROOT / 'data' / 'rgm_l3_k64.pt'))
    ap.add_argument('--rgm-levels', type=int, default=2, help='rgm: 1 = level-1 induction only, 2 = with events')
    ap.add_argument('--rgm-learn', action='store_true', help='rgm feedback: learn B1 counts from real transitions')
    ap.add_argument('--rgm-lr', type=float, default=1.0,
                    help='rgm learning: weight of one real in-episode transition, in units of offline Dirichlet counts')
    ap.add_argument('--bridge-hold', type=int, default=1,
                    help='rgm bridge: keep a decision for this many calls (1 = decide every call, 2 = per level-3 '
                         'action, 4 = per level-2 event); a surprising real step always forces a new decision')
    ap.add_argument('--bridge-f3', type=float, default=-1.0,
                    help='rgm bridge: F3 weight; -1 = learned precision (calibrated offline, updated from reality)')
    ap.add_argument('--rgm-every-action', action='store_true',
                    help='rgm feedback: replan (with the F3 check) after every real level-3 action, not only on surprise')
    ap.add_argument('--rgm-mode', default='closed', choices=['closed', 'commit', 'adaptive', 'feedback', 'bridge'],
                    help='rgm: closed = replan every call; commit = follow the route, replan on drift; '
                         'adaptive = EFE choice between continuing the route and switching to a fresh plan')
    ap.add_argument('--switch-cost', type=float, default=0.5, help='rgm adaptive: habit cost of switching (nats)')
    ap.add_argument('--tasks', type=int, nargs='+', default=list(range(13)))
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--env', default='ant', choices=['ant', 'pusht'], help='paper environment (eval config <env>_l<levels>)')
    ap.add_argument('--levels', type=int, default=3, choices=[2, 3], help='H-JEPA model depth')
    ap.add_argument('--sigma-goal', type=float, default=1.0)
    ap.add_argument('--mcts-repeats', type=int, default=300)
    ap.add_argument('--expl-scale', type=float, default=0.1)
    ap.add_argument('--plan-len', type=int, default=0, help='tschantz: plan this many level-3 steps internally (0 = H)')
    ap.add_argument('--route', action='store_true', help='tschantz: play in z3 until near the goal, H-JEPA follows the route, then finishes')
    ap.add_argument('--near', type=float, default=3.0, help='route: imagined state counts as at the goal (whitened distance)')
    ap.add_argument('--drift', type=float, default=6.0, help='route: re-play when the ant is this far from the route')
    ap.add_argument('--max-route', type=int, default=40, help='route: max imagined level-3 steps')
    ap.add_argument('--tag', default='')
    ap.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'l3_planner'))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.planner + a.tag
    if a.single:
        r = run_task(a, a.tasks[0])
        (out / f'{name}_task{a.tasks[0]:02d}.json').write_text(json.dumps(r, default=float))
        print(json.dumps({k: v for k, v in r.items() if k != 'decisions'}), flush=True)
        return
    env = dict(os.environ, PYTHONUTF8='1', HJEPA_HOME=os.environ.get('HJEPA_HOME', str(ROOT / 'data')),
               HOME=os.environ.get('HOME', os.environ.get('USERPROFILE', '')))
    for t in a.tasks:
        if (out / f'{name}_task{t:02d}.json').exists():
            continue
        cmd = [sys.executable, __file__, '--single', '--planner', a.planner, '--tasks', str(t), '--seed', str(a.seed), '--env', a.env, '--levels', str(a.levels),
               '--sigma-goal', str(a.sigma_goal), '--mcts-repeats', str(a.mcts_repeats), '--expl-scale', str(a.expl_scale), '--plan-len', str(a.plan_len), '--near', str(a.near), '--drift', str(a.drift), '--max-route', str(a.max_route), *(['--route'] if a.route else []),
               '--rgm-path', a.rgm_path, '--rgm-levels', str(a.rgm_levels), '--rgm-mode', a.rgm_mode,
               '--switch-cost', str(a.switch_cost), *(['--rgm-learn'] if a.rgm_learn else []),
               *(['--rgm-every-action'] if a.rgm_every_action else []),
               '--bridge-hold', str(a.bridge_hold), '--bridge-f3', str(a.bridge_f3), '--rgm-lr', str(a.rgm_lr),
               '--rgm-obs-level', str(a.rgm_obs_level), '--rgm-paper-path', a.rgm_paper_path,
               '--rgm-stuck', str(a.rgm_stuck), *(['--rgm-relax'] if a.rgm_relax else []),
               '--rgm-ctx-seqs', a.rgm_ctx_seqs, '--rgm-lookahead', str(a.rgm_lookahead),
               '--rgm-observe-every', str(a.rgm_observe_every),
               *(['--rgm-full'] if a.rgm_full else []), *(['--rgm-no-filter'] if a.rgm_no_filter else []),
               *(['--rgm-no-gate'] if a.rgm_no_gate else []), '--rgm-f3', a.rgm_f3,
               '--rgm-f3-weight', str(a.rgm_f3_weight), *(['--rgm-agent'] if a.rgm_agent else []),
               *(['--rgm-live'] if a.rgm_live else []), *(['--rgm-dream'] if a.rgm_dream else []),
               '--rgm-calib-seqs', a.rgm_calib_seqs, '--rgm-persist', a.rgm_persist, '--tag', a.tag, '--out', str(out)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        line = [l for l in r.stdout.splitlines() if l.startswith('{')]
        print(line[-1] if line else r.stderr[-2000:], flush=True)
    res = [json.loads((out / f'{name}_task{t:02d}.json').read_text()) for t in a.tasks if (out / f'{name}_task{t:02d}.json').exists()]
    print(f'{name}: {sum(r["success"] for r in res)}/{len(res)} tasks solved')


if __name__ == '__main__':
    main()
