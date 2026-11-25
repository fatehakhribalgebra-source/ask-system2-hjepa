"""One continuous life in the AntMaze: no episodes, no resets.

The ant is born at one benchmark start state with an EMPTY curious RGM over frozen H-JEPA. It first plays (explore
mode, no goal) for --play-steps env steps. Then a goal is thrown every --goal-every env steps (the benchmark's goal
images, in task order). The ant is never moved back: each goal starts from wherever the last one left it. When it
reaches a goal it plays again until the next goal arrives. The map is never reset either: every step of play and of
goal pursuit is experience (sliding-window transitions, states grown on the fly, dreams at each switch).

Output (--out): life.json (per-goal results, per-bout RGM stats, map summaries) and life_trace.npz (xy, phase and goal
index at every env step, plus the goal positions) for plotting the whole life.

  python experiments/life_run.py --seed 42 --play-steps 7000 --goals 20 --out results/life/seed42
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time
from copy import deepcopy

ROOT = pathlib.Path(__file__).resolve().parents[1]
HJ = ROOT / 'H-JEPA' / 'h_jepa'


def task_steps(world, ep, columns, start_mode, goal_mode):
    """Start and goal info of one benchmark task (the same steps World.evaluate_from_dataset takes)."""
    import numpy as np
    import torch
    init, goal = {}, {}
    for col in columns:
        if col.startswith('goal') or not isinstance(ep[col], (torch.Tensor, np.ndarray)):
            continue
        v = ep[col].permute(0, 2, 3, 1) if col.startswith('pixels') else ep[col]
        a0, a1 = v[0], v[-1]
        a0 = a0.numpy() if isinstance(a0, torch.Tensor) else a0
        a1 = a1.numpy() if isinstance(a1, torch.Tensor) else a1
        init[col] = np.stack([a0])
        goal['goal' if col == 'pixels' else f'goal_{col}'] = np.stack([a1])
    if 'step_idx' in init:
        init['step_idx'] = np.zeros_like(init['step_idx'])
    world._apply_xy_neutral_eval_state(init, goal, start_mode, goal_mode)
    return init, goal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--env', default='ant')
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--birth-task', type=int, default=0, help='benchmark task whose start state the ant is born in')
    ap.add_argument('--goal-tasks', type=int, nargs='+', default=None, help='goal order (default: tasks 0..goals-1)')
    ap.add_argument('--goals', type=int, default=20)
    ap.add_argument('--goal-every', type=int, default=350, help='env steps between thrown goals (the eval budget)')
    ap.add_argument('--play-steps', type=int, default=7000, help='env steps of play before the first goal')
    ap.add_argument('--brain', default='', help='start from this persisted CuriousRGM instead of an empty one')
    ap.add_argument('--explorer', default='curious', choices=['curious', 'random'])
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--no-dream', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
    os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'
    sys.path.insert(0, str(HJ)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import planning_eval as pe
    import stable_worldmodel as swm
    from stable_worldmodel.world import _find_env_method, _pad_loaded_eval_trajectories
    from eval_l3_planner import PaperLevel
    from system2.rgm_curious import CuriousRGM

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(HJ / 'config' / 'eval' / f'{a.env}_l{a.levels}.yaml')
    cfg.seed = a.seed
    name = f'{a.env}_hjepa_l{a.levels}'
    cfg.policy = str(ROOT / 'data' / 'ckpts' / a.env / name / f'seed{a.seed}' / f'{name}_object.ckpt')
    cfg.cache_dir = os.environ['HJEPA_HOME']
    budget = int(a.goal_every)
    cfg.eval.eval_budget = budget
    cfg.world.max_episode_steps = 10 ** 7                   # one life: never truncated
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    transform = {'pixels': pe.img_transform(cfg), 'goal': pe.img_transform(cfg)}
    process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
    policy = pe._build_policy(cfg, process, transform, model=None)
    policy.eval_budget = budget
    solver = policy.solver
    dev = 'cuda'

    # ---- the brain: curious RGM at level 3, H-JEPA levels 2 and 1 are the reflex
    if a.brain:
        with open(a.brain, 'rb') as fh:
            prgm = pickle.load(fh)
        prgm.reset()
    else:
        prgm = CuriousRGM(); prgm.rho = a.rho
    prgm.explorer, prgm.eta = a.explorer, a.eta
    prgm.rng = np.random.default_rng(a.seed)
    top = solver.level_models[len(solver.level_models) - 1]
    ln = list(top.action_encoder.children())[-1]
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        protos = ln(torch.randn(12, ln.normalized_shape[0], generator=g).to(dev))

    def f3_latents(zn, _top=top, _P=protos):
        with torch.no_grad():
            zz = torch.as_tensor(zn, dtype=torch.float32, device=dev).reshape(1, 1, -1).repeat(len(_P), 1, 1)
            return _top.predict(zz, _top.action_embed(_P[:, None]))[:, -1].float().cpu().numpy()
    prgm.f3_latents, prgm.pattern = f3_latents, None
    if not a.no_dream:
        gb = torch.Generator(device=dev).manual_seed(a.seed * 1000)

        def f3_batch(Zb, _top=top, _ln=ln, _g=gb):
            with torch.no_grad():
                u = torch.randn(len(Zb), _ln.normalized_shape[0], generator=_g, device=dev)
                zz = torch.as_tensor(Zb, dtype=torch.float32, device=dev)[:, None]
                return _top.predict(zz, _top.action_embed(_ln(u)[:, None]))[:, -1].float().cpu().numpy()
        prgm.dreaming, prgm.f3_batch = True, f3_batch
    level = PaperLevel(solver.level_solvers[3], prgm, None, 3, solver)
    solver.level_solvers[3] = level

    life = dict(t=0, phase='play', bout_first=None)
    trace = dict(xy=[], phase=[], goal=[])

    _orig_prep = policy._prepare_info

    def _prep(info_dict, _o=_orig_prep):
        if life['phase'] == 'play':                         # no goal: H-JEPA's own planner (on handover) is aimed
            if life['bout_first'] is None:                  # at where this play bout began, not at any task goal
                life['bout_first'] = {k: np.copy(info_dict[k]) for k in ('pixels', 'proprio') if k in info_dict}
            info_dict = dict(info_dict)
            info_dict['goal'] = life['bout_first']['pixels']
            if 'proprio' in life['bout_first']:
                info_dict['goal_proprio'] = life['bout_first']['proprio']
        o = _o(info_dict)
        with torch.no_grad():
            z3 = solver._encode_all_levels(o, key='pixels')['embed_3'][0, -1].float().cpu().numpy()
        prgm.record(life['t'], z3)                          # one life clock: the sliding window never restarts
        return o
    policy._prepare_info = _prep

    # ---- the world: born at one task's start, goals from the benchmark tasks
    payload = torch.load(str(HJ / cfg.load_eval_trajs_path), map_location='cpu', weights_only=False)
    columns = payload['columns']
    data = _pad_loaded_eval_trajectories([dict(d) for d in payload['data']], columns)
    goal_tasks = a.goal_tasks or list(range(a.goals))
    img = int(cfg.eval.img_size)
    world = swm.World(**{**world_cfg, 'num_envs': 1}, image_shape=(img, img))
    world.set_policy(policy)
    env = world.envs.unwrapped.envs[0]
    smode, gmode = cfg.eval.start_state_mode, cfg.eval.goal_state_mode
    ep0 = data[a.birth_task]
    seed0 = np.asarray([int(np.asarray(ep0['seed'][0]).reshape(-1)[0])]) if 'seed' in ep0 else None
    opts = [{}]
    var = {c.removeprefix('variation.'): np.asarray(ep0[c][0]) for c in columns if c.startswith('variation.')}
    if var:
        opts[0] = dict(variation=list(var), variation_values=var)
    world.reset(seed=seed0, options=opts)                   # env must exist before neutral states are rendered
    init, _ = task_steps(world, deepcopy(data[a.birth_task]), columns, smode, gmode)
    goals = [task_steps(world, deepcopy(data[t]), columns, smode, gmode)[1] for t in goal_tasks]
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    birth = {**init, **goals[0]}
    for spec in callables:                                  # place the ant (set_state); the goal is set per bout
        if spec['method'] != 'set_state':
            continue
        _find_env_method(env, 'set_state')(**{k: deepcopy(birth[v['value']][0]) for k, v in spec['args'].items()})
    shape = world.infos['pixels'].shape[:2]
    world.infos.update({k: np.broadcast_to(v[:, None, ...], shape + v.shape[1:]).copy() for k, v in init.items()})
    world.infos['goal'] = world.infos['pixels'].copy()      # play has no goal; the policy still expects the key
    goal_xy = np.stack([gs['goal_xy'][0] for gs in goals])

    def throw(gs):
        _find_env_method(env, 'set_navigation_goal')(goal_xy=deepcopy(gs['goal_xy'][0]))
        _find_env_method(env, 'set_current_goal_info')(**{k: deepcopy(gs[k][0]) for k in
                                                          ('goal', 'goal_xy', 'goal_qpos', 'goal_qvel', 'goal_proprio')
                                                          if k in gs})
        return {k: np.broadcast_to(v[:, None, ...], shape + v.shape[1:]).copy() for k, v in gs.items()}

    bouts, results = [], []

    def new_bout(phase, k=None):
        if level.decisions or life['t']:
            st = {kk: int(v) for kk, v in prgm.stats.items()}
            bouts.append(dict(phase=life['phase'], goal=life.get('k'), end=life['t'], stats=st,
                              decisions=len(level.decisions),
                              handovers=sum(1 for d in level.decisions if d.get('handover')),
                              map=prgm.map_summary()))
        keep = (getattr(prgm, '_stream', {}), getattr(prgm, '_visited', set()))
        prgm.reset()                                        # plan and counters only; the map is kept
        prgm._stream, prgm._visited = keep                  # the sliding window runs on across the switch
        prgm.explore = phase == 'play'
        policy.reset()
        level.__dict__['decisions'], level.__dict__['_target'] = [], None
        life.update(phase=phase, k=k, bout_first=None)

    def run(n, gstep, k=None):
        """n env steps; returns the step (1-based) the goal was first reached, or None."""
        hit = None
        for i in range(n):
            if life['phase'] == 'play' and policy._steps_taken >= budget:
                new_bout('play', k)                         # long play: a fresh bout every `budget` steps
            if gstep is not None:
                world.infos.update(deepcopy(gstep))
            world.step()
            world.envs.unwrapped._autoreset_envs = np.zeros((1,))
            life['t'] += 1
            trace['xy'].append(np.asarray(world.infos['xy']).reshape(-1)[:2].copy())
            trace['phase'].append(0 if life['phase'] == 'play' else 1)
            trace['goal'].append(-1 if k is None else k)
            if k is not None and hit is None and bool(np.asarray(world.terminateds).reshape(-1)[0]):
                hit = i + 1
                new_bout('play', k)                         # reached: play again until the next goal
        return hit

    t0 = time.time()
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        try:
            new_bout('play')
            run(a.play_steps, None)
            print(f'play done: {a.play_steps} steps, map {prgm.map_summary()}, {time.time() - t0:.0f}s', flush=True)
            for k, gs in enumerate(goals):
                gstep = throw(gs)
                new_bout('goal', k)
                d0 = float(np.linalg.norm(np.asarray(world.infos['xy']).reshape(-1)[:2] - goal_xy[k]))
                hit = run(budget, gstep, k)
                results.append(dict(goal=k, task=goal_tasks[k], success=hit is not None, steps=hit,
                                    start_dist=d0, t_thrown=life['t'] - budget))
                n_ok = sum(r['success'] for r in results)
                print(f'goal {k:2d} (task {goal_tasks[k]:2d}): {"reached in " + str(hit) if hit else "missed":>16}'
                      f' | start dist {d0:5.2f} | {n_ok}/{k + 1} | map K={prgm.K} | {time.time() - t0:.0f}s',
                      flush=True)
            new_bout('play')
        finally:
            world.close()
            np.savez_compressed(out / 'life_trace.npz', xy=np.array(trace['xy']), phase=np.array(trace['phase']),
                                goal=np.array(trace['goal']), goal_xy=goal_xy)
            (out / 'life.json').write_text(json.dumps(dict(args=vars(a), goals=results, bouts=bouts,
                                                           seconds=time.time() - t0), indent=1, default=str))
            with open(out / 'brain.pkl', 'wb') as fh:
                pickle.dump(prgm, fh)
    print(f'life over: {sum(r["success"] for r in results)}/{len(results)} goals reached', flush=True)


if __name__ == '__main__':
    main()
