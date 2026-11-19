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

    def __init__(self, orig, agent_fn, name):
        self.__dict__['_orig'] = orig
        self.__dict__['_agent_fn'] = agent_fn
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

    cfg = OmegaConf.load(HJ / 'config' / 'eval' / 'ant_l3.yaml')
    cfg.seed = a.seed
    cfg.policy = str(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / f'seed{a.seed}' / 'ant_hjepa_l3_object.ckpt')
    cfg.cache_dir = os.environ['HJEPA_HOME']
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    transform = {'pixels': pe.img_transform(cfg), 'goal': pe.img_transform(cfg)}
    process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
    policy = pe._build_policy(cfg, process, transform, model=None)
    solver = policy.solver
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values() if hasattr(s, 'torch_gen')]
    for gen, seed in gens:
        gen.manual_seed(seed + t)                           # identical to the paper's chunked eval

    dev = 'cuda'
    if a.planner != 'gd':
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

    payload = torch.load(str(HJ / cfg.load_eval_trajs_path), map_location='cpu', weights_only=False)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    chunk = {**payload, 'data': [dict(payload['data'][t])], 'episodes_idx': [t]}
    world = swm.World(**{**world_cfg, 'num_envs': 1}, image_shape=(64, 64))
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
    l3 = solver.level_solvers[3]
    r = dict(task=t, planner=a.planner, success=bool(m['episode_successes'][0]),
             steps_to_success=float(m['steps_to_success'][0]), seconds=time.time() - t0,
             decisions=getattr(l3, 'decisions', None))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--planner', default='gd', choices=['gd', 'fountas', 'tschantz'])
    ap.add_argument('--tasks', type=int, nargs='+', default=list(range(13)))
    ap.add_argument('--seed', type=int, default=42)
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
        cmd = [sys.executable, __file__, '--single', '--planner', a.planner, '--tasks', str(t), '--seed', str(a.seed),
               '--sigma-goal', str(a.sigma_goal), '--mcts-repeats', str(a.mcts_repeats), '--expl-scale', str(a.expl_scale), '--plan-len', str(a.plan_len), '--near', str(a.near), '--drift', str(a.drift), '--max-route', str(a.max_route), *(['--route'] if a.route else []), '--tag', a.tag, '--out', str(out)]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
        line = [l for l in r.stdout.splitlines() if l.startswith('{')]
        print(line[-1] if line else r.stderr[-2000:], flush=True)
    res = [json.loads((out / f'{name}_task{t:02d}.json').read_text()) for t in a.tasks if (out / f'{name}_task{t:02d}.json').exists()]
    print(f'{name}: {sum(r["success"] for r in res)}/{len(res)} tasks solved')


if __name__ == '__main__':
    main()
