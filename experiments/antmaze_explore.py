"""Reward-free exploration in Visual AntMaze (OGBench medium) with a frozen pretrained H-JEPA.

Conditions (--goal-source):
  pattern:<name>   System 2: pattern stream -> reservoir -> K latent candidates -> EFE score -> H-JEPA
                   (<name> e.g. fractal, fractal_shuffled, lorenz, levy, rule30, noise_labels, noise_gauss)
  random_latent    one Gaussian latent step per segment, no scoring (no pattern, no EFE)
  random_actions   uniform random torques (no world model)
Metric (Free Lunch style): % of maze cells visited over the episode, plus % of time in motion.

Usage:
  python experiments/antmaze_explore.py --goal-source pattern:fractal --episodes 3 --steps 1000
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

from system2.hjepa_bridge import HJEPA_DIR, TopLevelImagine, build_policy  # noqa: E402
from system2.goal_engine import EngineConfig, GoalEngine  # noqa: E402
from system2.patterns import make_stream  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.world import _find_env_method  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed{seed}' / 'ant_hjepa_l3_object.ckpt'


def loco_of(world):
    env = world.envs.unwrapped.envs[0]
    return _find_env_method(env, 'neutral_info_from_xy').__self__


class Recorder:
    def __init__(self, loco, scale=8):
        self.loco, self.scale = loco, scale
        self.frames, self.traj, self.goal_xy = [], [], []
        mm = np.asarray(loco.env.maze_map)
        self.mm = mm
        self.base = np.where(mm[..., None] == 1, 40, 230).astype(np.uint8).repeat(3, -1)

    def ij(self, xy):
        return np.asarray(self.loco.env.xy_to_ij(xy))

    def frame(self, pixels, xy, goal_xy=None, text=None):
        import cv2
        s = 32
        mp = cv2.resize(self.base, (self.mm.shape[1] * s, self.mm.shape[0] * s), interpolation=cv2.INTER_NEAREST)
        u = float(self.loco.env._maze_unit)
        ox, oy = float(self.loco.env._offset_x), float(self.loco.env._offset_y)
        to_px = lambda p: (int((p[0] + ox) / u * s + s / 2), int((p[1] + oy) / u * s + s / 2))
        self.traj.append(to_px(xy))
        for a, b in zip(self.traj[:-1], self.traj[1:]):
            cv2.line(mp, a, b, (0, 160, 0), 2)
        if goal_xy is not None:
            cv2.drawMarker(mp, to_px(goal_xy), (220, 0, 220), cv2.MARKER_STAR, 18, 2)
        cv2.circle(mp, to_px(xy), 7, (0, 0, 255), -1)
        cam = cv2.resize(pixels, (mp.shape[0], mp.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = np.concatenate([cam, mp], axis=1)
        if text:
            cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        self.frames.append(out)

    def save(self, path):
        import imageio
        imageio.mimsave(path, self.frames, fps=20, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-source', default='pattern:fractal')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--segment', type=int, default=100, help='env steps per System-2 goal')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--imagine-h', type=int, default=5, help='top-level macro-steps imagined per goal')
    ap.add_argument('--act-scale', type=float, default=1.0)
    ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--fast', action='store_true', help='fewer planner samples (quicker, weaker)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze'))
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.goal_source.replace(':', '_')
    overrides = {'eval.eval_budget': a.segment}
    if a.fast:
        overrides.update({'solver.solvers.level3.num_samples': 100, 'solver.solvers.level1.n_steps': 40,
                          'solver.solvers.level2.n_steps': 40, 'solver.solvers.level3.n_steps': 40})
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'ant_l3', overrides=overrides)
    L = model.num_levels
    top = solver.level_models[L - 1]
    D = None

    cfg.world.max_episode_steps = a.steps + 10
    cfg.world.num_envs = 1
    from omegaconf import OmegaConf
    wcfg = OmegaConf.to_container(cfg.world, resolve=True); wcfg.update(num_envs=1, max_episode_steps=a.steps + 10, terminate_at_goal=False,
                                        dataset_type='explore')
    world = swm.World(**wcfg, image_shape=(64, 64), verbose=0)
    world.set_policy(policy)
    loco = loco_of(world)
    cells = {tuple(c) for c in loco.all_cells}

    results = []
    engine = None
    for ep in range(a.episodes):
        world.reset(seed=a.seed * 1000 + ep)
        rec = Recorder(loco) if (a.record and ep == 0) else None
        visited, moving, prev_xy = set(), 0, None
        xy_z = []                    # (z_top, xy) pairs for a linear probe (visualisation only)
        goal_xy_est = None
        seg_log = []
        t0 = time.time()
        for t in range(a.steps):
            infos = world.infos
            xy = np.asarray(infos['xy'][0]).reshape(-1)[-2:]
            visited.add(tuple(np.asarray(loco.env.xy_to_ij(xy)).tolist()))
            if prev_xy is not None:
                moving += float(np.linalg.norm(xy - prev_xy) > 0.05)
            prev_xy = xy

            if a.goal_source == 'random_actions':
                act = world.envs.action_space.sample()
                world.states, world.rewards, world.terminateds, world.truncateds, world.infos = world.envs.step(act)
                if rec: rec.frame(np.asarray(infos['pixels'][0]).reshape(64, 64, 3), xy, None, f'random t={t}')
                continue

            if engine is not None and t % 20 == 0 and t % a.segment != 0:
                info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos.items()}
                with torch.no_grad():
                    zz = solver._encode_all_levels(policy._prepare_info(info), key='pixels')[f'embed_{L}']
                engine.observe(zz.reshape(-1))                 # episodic memory for the novelty term
            if t % a.segment == 0:
                # --- encode current state at the top level
                info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos.items()}
                prep = policy._prepare_info(info)
                with torch.no_grad():
                    z = solver._encode_all_levels(prep, key='pixels')[f'embed_{L}'].reshape(-1)
                D = z.numel()
                xy_z.append((z.cpu().numpy(), xy.copy()))
                kind, _, pname = a.goal_source.partition(':')
                if engine is None and kind in ('pattern', 'direct'):
                    stream = make_stream(pname, seed=a.seed)
                    k = a.k if kind == 'pattern' else 1
                    engine = GoalEngine(D, stream, EngineConfig(k=k, learn_readout=False), seed=a.seed)
                    engine.init_imagination(horizon=a.imagine_h, action_dim=4, act_scale=a.act_scale)
                    imagine = TopLevelImagine(top)
                if engine is not None:
                    if t > 0:
                        engine.update(z)
                        engine.ens.fit(steps=20, device=z.device)
                    else:
                        engine.reset(episode_offset=ep * 37)
                    engine.observe(z)
                    goal, info_s = engine.select_imagined(z, imagine, use_efe=(kind == 'pattern'))
                    sel = dict(G=float(info_s['G'][info_s['k']]), epi=float(info_s['epi'][info_s['k']]),
                               nov=float(info_s['nov'][info_s['k']]))
                elif kind == 'random_macro':
                    if 'imagine' not in dir():
                        imagine = TopLevelImagine(top)
                    acts = a.act_scale * torch.randn(1, a.imagine_h, 4, device=z.device)
                    goal = imagine(z, acts)[0]; sel = {}
                else:
                    raise ValueError(a.goal_source)
                solver.goal_top = goal.reshape(1, 1, -1).to(z.device)
                policy.reset()                      # restart the per-goal horizon schedule
                # probe: where in the maze does the selected latent goal correspond to? (viz only)
                if len(xy_z) >= 6:
                    Z = np.stack([p[0] for p in xy_z]); Y = np.stack([p[1] for p in xy_z])
                    Zm = Z.mean(0); W = np.linalg.solve((Z - Zm).T @ (Z - Zm) + 10 * np.eye(D), (Z - Zm).T @ (Y - Y.mean(0)))
                    goal_xy_est = (goal.detach().cpu().numpy() - Zm) @ W + Y.mean(0)
                seg_log.append(dict(t=t, visited=len(visited), goal_dist=float((goal - z).norm()), **sel))

            raw_px = np.asarray(infos['pixels']).reshape(-1, 64, 64, 3)[-1].copy()   # before policy mutates infos
            infos['goal'] = infos['pixels']            # placeholder; the solver ignores goal pixels
            world.step()
            if rec:
                rec.frame(raw_px.astype(np.uint8), xy, goal_xy_est, f'{a.goal_source} t={t} cells={len(visited)}')

        cov = len(visited & cells) / len(cells)
        r = dict(episode=ep, coverage=cov, cells=len(visited & cells), total_cells=len(cells),
                 motion=moving / max(1, a.steps - 1), seconds=time.time() - t0, segments=seg_log)
        results.append(r)
        print(f'[{a.goal_source}] ep {ep}: coverage {100 * cov:.1f}% ({r["cells"]}/{len(cells)} cells), '
              f'motion {100 * r["motion"]:.0f}%, {r["seconds"]:.0f}s', flush=True)
        if rec:
            rec.save(out / f'replay_{tag}_seed{a.seed}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1,
                                                              default=float))


if __name__ == '__main__':
    main()
