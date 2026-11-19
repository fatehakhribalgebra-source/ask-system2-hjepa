"""Reward-free exploration in Visual AntMaze with System 2 v3 (continuous pattern drive, learning readout)
over the frozen pretrained 3-level H-JEPA (the paper model, reproduced at 69% goal success on this machine).

Conditions (--source): fractal, fractal_shuffled, noise_gauss, levy, lorenz, ... (any make_stream name),
or 'none' (no reservoir: i.i.d. random macro-actions, same EFE choice). --no-learn freezes the readout.
The readout (and ensemble) persist across episodes, so later episodes show what was learned.
Metric: % of maze cells visited (true xy, never seen by the agent), coverage-vs-steps curve, % time in motion.
One World per process (Windows/glfw renders black frames in a second World).

  python experiments/antmaze_explore_v3.py --source fractal --episodes 5 --steps 1000
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))

from system2.goal_engine_v3 import GoalEngineV3, V3Config  # noqa: E402
from system2.hjepa_bridge import TopLevelImagine, build_policy  # noqa: E402
from system2.patterns import make_stream  # noqa: E402
sys.path.insert(0, str(ROOT / 'experiments'))
from antmaze_explore import Recorder  # noqa: E402  (camera | whole-maze map with path, as the v1 replays)

import stable_worldmodel as swm  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from stable_worldmodel.world import _find_env_method  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed{seed}' / 'ant_hjepa_l3_object.ckpt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='fractal')
    ap.add_argument('--no-learn', action='store_true')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--check', type=int, default=25, help='env steps between commitment checks')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--sigma', type=float, default=0.5)
    ap.add_argument('--generator', default='reservoir', choices=['reservoir', 'vae', 'cem', 'efe'],
                    help='reservoir = v3; vae = pattern code -> macro-action VAE decoder (G); cem = CEM on EFE (C); efe = VAE proposals scored by real EFE')
    ap.add_argument('--w-ig', type=float, default=1.0); ap.add_argument('--w-amb', type=float, default=1.0)
    ap.add_argument('--w-risk', type=float, default=1.0); ap.add_argument('--w-prior', type=float, default=1.0)
    ap.add_argument('--efe-tag', default='', help='suffix for ablation runs')
    ap.add_argument('--efe-dims', type=int, default=32, help='whitened PCA dims of the latent the EFE model observes')
    ap.add_argument('--vae', default=str(ROOT / 'data' / 'action_vae_ant_l3.pt'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--full-planner', action='store_true', help='paper planner budget (slower); default = --fast')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze_v3'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.source}_{'frozen' if a.no_learn else 'learn'}" if a.generator == 'reservoir' else (f'vae_{a.source}' if a.generator == 'vae' else ('cem' if a.generator == 'cem' else f'efe_{a.source}{a.efe_tag}'))

    ov = {'eval.eval_budget': 8 * a.check}     # > longest commitment (max_age 6 checks), reset at every new goal
    if not a.full_planner:
        ov.update({'solver.solvers.level3.num_samples': 100, **{f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)}})
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'ant_l3', overrides=ov)
    L = model.num_levels
    imagine = TopLevelImagine(solver.level_models[L - 1])
    wcfg = OmegaConf.to_container(cfg.world, resolve=True)
    wcfg.update(num_envs=1, max_episode_steps=a.steps + 10, terminate_at_goal=False, dataset_type='explore')
    world = swm.World(**wcfg, image_shape=(64, 64), verbose=0)
    world.set_policy(policy)
    loco = _find_env_method(world.envs.unwrapped.envs[0], 'neutral_info_from_xy').__self__
    cells = {tuple(c) for c in loco.all_cells}

    stream = None if a.source in ('none',) else make_stream(a.source, seed=a.seed)
    engine = None
    results = []
    for ep in range(a.episodes):
        world.reset(seed=a.seed * 1000 + ep)
        visited, moving, prev, traj, seg_log, curve, t0 = set(), 0, None, [], [], [], time.time()
        upright, first_flip = 0, None
        rec = Recorder(loco) if (a.record and ep == a.episodes - 1) else None
        z, xy_z, goal_xy, model_losses = None, [], None, []
        for t in range(a.steps):
            infos = world.infos
            xy = np.asarray(infos['xy'][0]).reshape(-1)[-2:].copy()
            visited.add(tuple(np.asarray(loco.env.xy_to_ij(xy)).tolist()))
            if prev is not None:
                moving += float(np.linalg.norm(xy - prev) > 0.05)
            prev = xy
            q = np.asarray(infos['qpos']).reshape(-1)[-15:]      # ant qpos: x y z qw qx qy qz + 8 joints
            up = float(q[3] ** 2 + q[6] ** 2 - q[4] ** 2 - q[5] ** 2)   # cos(torso tilt): 1 upright, <0 upside down
            upright += up > 0.0
            if up <= 0.0 and first_flip is None:
                first_flip = t
            if t % 2 == 0:
                traj.append(xy.round(2).tolist())
            if t % 100 == 0:
                curve.append(len(visited & cells))
            if t % a.check == 0:
                info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos.items()}
                with torch.no_grad():
                    z = solver._encode_all_levels(policy._prepare_info(info), key='pixels')[f'embed_{L}'].reshape(-1)
                if engine is None and a.generator == 'vae':
                    from system2.action_vae import ActionVAE
                    from system2.goal_engine_gen import GoalEngineGen
                    ck = torch.load(a.vae, map_location='cuda', weights_only=False)
                    vae = ActionVAE(ck['d_z'], ck['H'], ck['A'], ck['d_u']).cuda().eval()
                    vae.load_state_dict(ck['state_dict'])
                    engine = GoalEngineGen(z.numel(), stream, vae, V3Config(k=a.k, horizon=ck['H']), seed=a.seed)
                if engine is None and a.generator == 'efe':
                    from system2.action_vae import ActionVAE
                    from system2.goal_engine_efe import EFEWeights, GoalEngineEFE
                    from system2.latent_dynamics import GaussEnsemble, MacroActionTracker
                    ck = torch.load(a.vae, map_location='cuda', weights_only=False)
                    vae = ActionVAE(ck['d_z'], ck['H'], ck['A'], ck['d_u']).cuda().eval()
                    vae.load_state_dict(ck['state_dict'])
                    Zdata = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')['z']
                    pmu, P = GaussEnsemble.pca_whitener(Zdata, k=a.efe_dims)            # observation = whitened PCA latent
                    dyn = GaussEnsemble(z.numel(), ck['A'], z_mean=pmu, proj=P).cuda()
                    engine = GoalEngineEFE(z.numel(), stream, vae, dyn, V3Config(k=a.k, horizon=ck['H']),
                                           EFEWeights(a.w_ig, a.w_amb, a.w_risk, a.w_prior), seed=a.seed)
                    nstats = torch.load(pathlib.Path(str(CKPT).format(seed=a.model_seed)).parent / 'normalizer.pt',
                                        weights_only=False)['stats']
                    tracker = MacroActionTracker(model, nstats)
                if engine is None and a.generator == 'cem':
                    from system2.goal_engine_cem import GoalEngineCEM
                    ck = torch.load(a.vae, map_location='cuda', weights_only=False)
                    a_std = ck['state_dict']['a_std'].reshape(ck['H'], ck['A']).mean(0)   # true macro-action spread
                    engine = GoalEngineCEM(z.numel(), V3Config(k=a.k, horizon=ck['H']), seed=a.seed, init_std=a_std)
                if engine is None:
                    engine = GoalEngineV3(z.numel(), stream, V3Config(
                        k=a.k, sigma=a.sigma, learn_readout=not a.no_learn, random_actions=stream is None), seed=a.seed)
                if t == 0:
                    engine.reset(episode_offset=ep * 37)
                    if a.generator == 'efe':
                        tracker.reset()
                engine.observe(z)
                if engine.should_reselect(z):
                    if engine.goal is not None:
                        engine.finish_goal(z)
                    goal, s = engine.select(z, imagine)
                    solver.goal_top = goal.reshape(1, 1, -1)
                    policy.reset()
                    seg_log.append(dict(t=t, cells=len(visited & cells), **s))
                xy_z.append((z.cpu().numpy(), xy.copy()))
                if engine.goal is not None and len(xy_z) >= 6:     # linear probe z -> xy, for the replay star only
                    Z = np.stack([q[0] for q in xy_z]); Y = np.stack([q[1] for q in xy_z]); Zm, Ym = Z.mean(0), Y.mean(0)
                    Wp = np.linalg.solve((Z - Zm).T @ (Z - Zm) + 10 * np.eye(Z.shape[1]), (Z - Zm).T @ (Y - Ym))
                    goal_xy = (engine.goal['goal'].cpu().numpy() - Zm) @ Wp + Ym
            engine.drive(z)
            if rec is not None:
                rec.frame(np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy().astype(np.uint8), xy, goal_xy,
                          f'{tag} ep{ep} t={t} cells={len(visited & cells)}/{len(cells)}')
            infos['goal'] = infos['pixels']       # placeholder; ExplorationSolver uses solver.goal_top
            if a.generator == 'efe':              # = world.step(), but keep the executed torques for exact learning
                px0 = np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy()
                qp0, qv0 = np.asarray(infos['qpos']).reshape(-1)[-15:].copy(), np.asarray(infos['qvel']).reshape(-1)[-14:].copy()
                act = world.policy.get_action(world.infos)
                world.states, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(act)
                tracker.push(px0, qp0, qv0, act)
                if (t + 1) % 20 == 0:
                    tr = tracker.transition()
                    if tr is not None:
                        lv = engine.learn_transition(*tr)
                        if lv is not None:
                            model_losses.append(lv)
            else:
                world.step()
        if engine.goal is not None:
            engine.finish_goal(z)
        cov = len(visited & cells) / len(cells)
        r = dict(episode=ep, coverage=cov, cells=len(visited & cells), total_cells=len(cells), curve=curve,
                 motion=moving / max(1, a.steps - 1), upright=upright / a.steps, first_flip=first_flip, seconds=time.time() - t0, goals=len(seg_log),
                 readout_updates=engine.stats['updates'], imagined_updates=engine.stats.get('imagined_updates', 0),
                 readout_norm=float(engine.W.norm()), segments=seg_log, traj=traj,
                 model_transitions=getattr(engine, 'n_trans', 0), model_loss=(float(np.mean(model_losses[-10:])) if model_losses else None))
        results.append(r)
        print(f'[{tag}] ep {ep}: coverage {100 * cov:.1f}% ({r["cells"]}/{len(cells)}), goals {r["goals"]}, '
              f'readout updates so far {r["readout_updates"]} real + {r["imagined_updates"]} imagined, motion {100 * r["motion"]:.0f}%, upright {100 * r["upright"]:.0f}% (flip at {r["first_flip"]}), {r["seconds"]:.0f}s',
              flush=True)
        if rec is not None:
            rec.frames = rec.frames[::2]
            rec.save(out / f'replay_{tag}_seed{a.seed}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1,
                                                             default=float))


if __name__ == '__main__':
    main()
