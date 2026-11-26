"""Life v2 (ASK Animal design, docs/DESIGN_ANIMAL.md, experiment E1): one continuous life with sleep.

Differences from experiments/life_run.py (v1):
- Two brains. Play uses CuriousRGM for exploration only, with no dreaming by default (R1: imagination counted as
  experience poisons plans). Goals use a map built only from what really happened.
- Sleep at every goal throw: the life's own real z3 stream (every env step so far, play and goals alike) is cut into
  20 sliding-window sequences (offset o: z3[o], z3[o+20], ...) and a paper-style RGM is fitted to it (whitening, k-means
  K, counted items: the same fit as the 24 Nov 13/13 map, experiments/build_rgm_stream.py), now inside the life.
- Goal planner: PaperRGM (fewest hops; no path -> H-JEPA plans) or RiskRGM (reach probability on seen counts; below
  --handover-v -> H-JEPA plans). Either way the RGM hands over when it is unsure (R2).
- The trace also logs torso height (qpos[2]) when the env exposes it, to tell stuck from flipped (R7).

  python experiments/life_v2.py --seed 42 --play-steps 7000 --goals 20 --out results/life_v2/seed42
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
sys.path.insert(0, str(ROOT / 'experiments'))
from life_run import task_steps  # noqa: E402

STRIDE = 20   # env steps per level-3 step


def sleep_fit(Z, k, seed, planner, lam, handover_v):
    """Refit the place map from the real z3 stream (one array per env step, in order)."""
    import numpy as np
    from system2.rgm_paper import PaperRGM
    Z = np.asarray(Z, np.float32)
    seqs, eps = [], []
    for o in range(STRIDE):                                   # sliding window: 20 sequences, one per offset
        idx = np.arange(o, len(Z), STRIDE)
        if len(idx) >= 2:
            seqs.append(Z[idx]); eps.append(np.full(len(idx), o))
    m = PaperRGM.fit(np.concatenate(seqs), np.concatenate(eps), k=k, seed=seed)
    if planner == 'risk':
        from system2.rgm_risk import RiskRGM
        m = RiskRGM.from_state(m.state_dict())
        m.time_discount, m.handover_v = lam, handover_v
        m._refresh()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--env', default='ant')
    ap.add_argument('--levels', type=int, default=3)
    ap.add_argument('--birth-task', type=int, default=0)
    ap.add_argument('--goal-tasks', type=int, nargs='+', default=None)
    ap.add_argument('--goals', type=int, default=20)
    ap.add_argument('--goal-every', type=int, default=350)
    ap.add_argument('--play-steps', type=int, default=7000)
    ap.add_argument('--explorer', default='curious', choices=['curious', 'random'])
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--rho', type=float, default=0.5)
    ap.add_argument('--dream', action='store_true', help='let the play brain dream (off by default: R1)')
    ap.add_argument('--sleep-k', type=int, default=64, help='places in the sleep-fitted map')
    ap.add_argument('--planner', default='paper', choices=['paper', 'risk'])
    ap.add_argument('--risk-lambda', type=float, default=0.8)
    ap.add_argument('--handover-v', type=float, default=0.0, help='risk planner: hand over below this reach prob.')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
    os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'
    sys.path.insert(0, str(HJ)); sys.path.insert(0, str(ROOT))
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
    cfg.world.max_episode_steps = 10 ** 7
    world_cfg = OmegaConf.create(OmegaConf.to_container(cfg.world, resolve=True))
    transform = {'pixels': pe.img_transform(cfg), 'goal': pe.img_transform(cfg)}
    process = pe._build_eval_process_from_policy_normalizer(cfg, model=None)
    policy = pe._build_policy(cfg, process, transform, model=None)
    policy.eval_budget = budget
    solver = policy.solver
    dev = 'cuda'

    # ---- play brain: curious RGM, exploration only
    play = CuriousRGM(); play.rho = a.rho
    play.explorer, play.eta, play.explore = a.explorer, a.eta, True
    play.rng = np.random.default_rng(a.seed)
    top = solver.level_models[len(solver.level_models) - 1]
    ln = list(top.action_encoder.children())[-1]
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        protos = ln(torch.randn(12, ln.normalized_shape[0], generator=g).to(dev))

    def f3_latents(zn, _top=top, _P=protos):
        with torch.no_grad():
            zz = torch.as_tensor(zn, dtype=torch.float32, device=dev).reshape(1, 1, -1).repeat(len(_P), 1, 1)
            return _top.predict(zz, _top.action_embed(_P[:, None]))[:, -1].float().cpu().numpy()
    play.f3_latents, play.pattern = f3_latents, None
    if a.dream:
        gb = torch.Generator(device=dev).manual_seed(a.seed * 1000)

        def f3_batch(Zb, _top=top, _ln=ln, _g=gb):
            with torch.no_grad():
                u = torch.randn(len(Zb), _ln.normalized_shape[0], generator=_g, device=dev)
                zz = torch.as_tensor(Zb, dtype=torch.float32, device=dev)[:, None]
                return _top.predict(zz, _top.action_embed(_ln(u)[:, None]))[:, -1].float().cpu().numpy()
        play.dreaming, play.f3_batch = True, f3_batch
    level = PaperLevel(solver.level_solvers[3], play, None, 3, solver)
    solver.level_solvers[3] = level

    life = dict(t=0, phase='play', bout_first=None, k=None)
    trace = dict(xy=[], phase=[], goal=[], height=[])
    stream_z = []                                             # every real z3 of the life, one per env step
    sleeps = []

    _orig_prep = policy._prepare_info

    def _prep(info_dict, _o=_orig_prep):
        if life['phase'] == 'play':                           # no goal: H-JEPA (on handover) aims at the bout start
            if life['bout_first'] is None:
                life['bout_first'] = {k: np.copy(info_dict[k]) for k in ('pixels', 'proprio') if k in info_dict}
            info_dict = dict(info_dict)
            info_dict['goal'] = life['bout_first']['pixels']
            if 'proprio' in life['bout_first']:
                info_dict['goal_proprio'] = life['bout_first']['proprio']
        o = _o(info_dict)
        with torch.no_grad():
            z3 = solver._encode_all_levels(o, key='pixels')['embed_3'][0, -1].float().cpu().numpy()
        play.record(life['t'], z3)                            # the play map keeps its sliding window
        if len(stream_z) == life['t']:                        # one latent per env step (first call wins)
            stream_z.append(z3)
        return o
    policy._prepare_info = _prep

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
    world.reset(seed=seed0, options=opts)
    init, _ = task_steps(world, deepcopy(data[a.birth_task]), columns, smode, gmode)
    goals = [task_steps(world, deepcopy(data[t]), columns, smode, gmode)[1] for t in goal_tasks]
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)
    birth = {**init, **goals[0]}
    for spec in callables:
        if spec['method'] != 'set_state':
            continue
        _find_env_method(env, 'set_state')(**{k: deepcopy(birth[v['value']][0]) for k, v in spec['args'].items()})
    shape = world.infos['pixels'].shape[:2]
    world.infos.update({k: np.broadcast_to(v[:, None, ...], shape + v.shape[1:]).copy() for k, v in init.items()})
    world.infos['goal'] = world.infos['pixels'].copy()
    goal_xy = np.stack([gs['goal_xy'][0] for gs in goals])

    def throw(gs):
        _find_env_method(env, 'set_navigation_goal')(goal_xy=deepcopy(gs['goal_xy'][0]))
        _find_env_method(env, 'set_current_goal_info')(**{k: deepcopy(gs[k][0]) for k in
                                                          ('goal', 'goal_xy', 'goal_qpos', 'goal_qvel', 'goal_proprio')
                                                          if k in gs})
        return {k: np.broadcast_to(v[:, None, ...], shape + v.shape[1:]).copy() for k, v in gs.items()}

    bouts, results = [], []

    def new_bout(phase, k=None, brain=None):
        if level.decisions or life['t']:
            cur = level._rgm
            st = {kk: (int(v) if isinstance(v, (int, float, np.integer)) else v) for kk, v in cur.stats.items()}
            bouts.append(dict(phase=life['phase'], goal=life.get('k'), end=life['t'], stats=st,
                              decisions=len(level.decisions),
                              handovers=sum(1 for d in level.decisions if d.get('handover')),
                              map=play.map_summary() if cur is play else dict(K=cur.K)))
        if brain is None:                                     # play: back to the curious brain, its map kept
            keep = (getattr(play, '_stream', {}), getattr(play, '_visited', set()))
            play.reset()
            play._stream, play._visited = keep
            play.explore = True
            level.__dict__['_rgm'], level.__dict__['_ex'] = play, None
        else:                                                 # goal: the brain fitted in this sleep
            brain.reset()
            level.__dict__['_rgm'] = brain
            level.__dict__['_ex'] = torch.as_tensor(brain.ex, dtype=torch.float32, device=dev)
        policy.reset()
        level.__dict__['decisions'], level.__dict__['_target'] = [], None
        life.update(phase=phase, k=k, bout_first=None)

    def run(n, gstep, k=None, brain=None):
        hit = None
        for i in range(n):
            if life['phase'] == 'play' and policy._steps_taken >= budget:
                new_bout('play', k)
            if gstep is not None:
                world.infos.update(deepcopy(gstep))
            world.step()
            world.envs.unwrapped._autoreset_envs = np.zeros((1,))
            life['t'] += 1
            trace['xy'].append(np.asarray(world.infos['xy']).reshape(-1)[:2].copy())
            q = world.infos.get('qpos') if hasattr(world.infos, 'get') else None
            trace['height'].append(float(np.asarray(q).reshape(-1)[2]) if q is not None and np.asarray(q).size > 2
                                   else np.nan)
            trace['phase'].append(0 if life['phase'] == 'play' else 1)
            trace['goal'].append(-1 if k is None else k)
            if k is not None and hit is None and bool(np.asarray(world.terminateds).reshape(-1)[0]):
                hit = i + 1
                life['goal_decisions'] = list(level.decisions)   # keep them: new_bout clears the level's list
                new_bout('play', k)                           # reached: play again until the next goal
        return hit

    t0 = time.time()
    with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
        try:
            new_bout('play')
            run(a.play_steps, None)
            print(f'play done: {a.play_steps} steps, play map {play.map_summary()}, {time.time() - t0:.0f}s',
                  flush=True)
            for k, gs in enumerate(goals):
                ts = time.time()
                brain = sleep_fit(stream_z, a.sleep_k, a.seed, a.planner, a.risk_lambda, a.handover_v)
                sleeps.append(dict(goal=k, latents=len(stream_z), K=brain.K, seconds=round(time.time() - ts, 1)))
                gstep = throw(gs)
                life['goal_decisions'] = None
                new_bout('goal', k, brain=brain)
                d0 = float(np.linalg.norm(np.asarray(world.infos['xy']).reshape(-1)[:2] - goal_xy[k]))
                hit = run(budget, gstep, k, brain=brain)
                dec = life['goal_decisions'] if life['goal_decisions'] is not None else list(level.decisions)
                own = sum(1 for d in dec if not d.get('handover'))
                results.append(dict(goal=k, task=goal_tasks[k], success=hit is not None, steps=hit, start_dist=d0,
                                    t_thrown=life['t'] - budget, rgm_decides=round(own / max(len(dec), 1), 3),
                                    sleep_latents=len(stream_z)))
                n_ok = sum(r['success'] for r in results)
                print(f'goal {k:2d} (task {goal_tasks[k]:2d}): {"reached in " + str(hit) if hit else "missed":>16}'
                      f' | start dist {d0:5.2f} | RGM decides {100 * own / max(len(dec), 1):3.0f}%'
                      f' | {n_ok}/{k + 1} | sleep {sleeps[-1]["seconds"]}s | {time.time() - t0:.0f}s', flush=True)
            new_bout('play')
        finally:
            world.close()
            np.savez_compressed(out / 'life_trace.npz', xy=np.array(trace['xy']), phase=np.array(trace['phase']),
                                goal=np.array(trace['goal']), height=np.array(trace['height']), goal_xy=goal_xy)
            (out / 'life.json').write_text(json.dumps(dict(args=vars(a), goals=results, bouts=bouts, sleeps=sleeps,
                                                           seconds=time.time() - t0), indent=1, default=str))
            np.save(out / 'stream_z3.npy', np.asarray(stream_z, np.float16))
            with open(out / 'play_brain.pkl', 'wb') as fh:
                pickle.dump(play, fh)
    print(f'life over: {sum(r["success"] for r in results)}/{len(results)} goals reached', flush=True)


if __name__ == '__main__':
    main()
