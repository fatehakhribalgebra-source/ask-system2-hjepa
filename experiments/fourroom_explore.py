"""Reward-free exploration in FourRoom Distractors with the frozen pretrained 3-level H-JEPA.

Conditions (--goal-source):
  pattern:<name>  System 2: pattern -> reservoir -> K imagined top-level goals -> EFE (novelty+epistemic) -> H-JEPA
  direct:<name>   same but K=1, no EFE selection (Free-Lunch-style "pattern drives behaviour")
  random_macro    random abstract macro-actions -> imagined goal -> H-JEPA
  random_actions  uniform random 2-d moves (no model)
Metrics: % of free arena cells visited (16x16 grid), rooms visited, % time in motion.
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

from system2.goal_engine import EngineConfig, GoalEngine  # noqa: E402
from system2.hjepa_bridge import TopLevelImagine, build_policy  # noqa: E402
from system2.patterns import make_stream  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'fourroom' / 'fourroom_hjepa_l3' / 'seed{seed}' / 'fourroom_hjepa_l3_object.ckpt'
G = 16                      # coverage grid
CELL = 224 / G


def inner_env(world):
    e = world.envs.unwrapped.envs[0]
    while hasattr(e, 'env') and not hasattr(e, '_position_is_free'):
        e = e.env
    return e.unwrapped if not hasattr(e, '_position_is_free') else e


def free_cells(env) -> set:
    """Cells containing any free position (sampled on a 3x3 sub-grid), so corridor/door cells count."""
    cells = set()
    for i in range(G):
        for j in range(G):
            for u in (0.15, 0.5, 0.85):
                for v in (0.15, 0.5, 0.85):
                    c = np.array([(j + u) * CELL, (i + v) * CELL], dtype=np.float32)
                    if env._position_is_free(c):
                        cells.add((i, j))
    return cells


def cell_of(xy):
    return int(np.clip(xy[1] // CELL, 0, G - 1)), int(np.clip(xy[0] // CELL, 0, G - 1))


def room_of(xy):
    return int(xy[1] > 112) * 2 + int(xy[0] > 112)


class Recorder:
    def __init__(self):
        self.frames, self.traj = [], []

    def frame(self, px, xy, goal_xy, text):
        import cv2
        img = cv2.resize(px, (448, 448), interpolation=cv2.INTER_NEAREST).copy()
        self.traj.append((int(xy[0] * 2), int(xy[1] * 2)))
        for a, b in zip(self.traj[:-1], self.traj[1:]):
            cv2.line(img, a, b, (0, 200, 0), 2)
        if goal_xy is not None and np.all(np.isfinite(goal_xy)):
            cv2.drawMarker(img, (int(goal_xy[0] * 2), int(goal_xy[1] * 2)), (230, 0, 230), cv2.MARKER_STAR, 22, 2)
        cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        self.frames.append(img)

    def save(self, path):
        import imageio
        imageio.mimsave(path, self.frames, fps=20, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-source', default='pattern:fractal')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--segment', type=int, default=60, help='env steps per System-2 goal')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--imagine-h', type=int, default=3)
    ap.add_argument('--act-scale', type=float, default=1.0)
    ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--fast', action='store_true', help='lighter planner (same for every condition)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'fourroom'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.goal_source.replace(':', '_')
    kind, _, pname = a.goal_source.partition(':')

    ov = {'eval.eval_budget': a.segment}
    if a.fast:
        ov.update({f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)})
        ov.update({'solver.solvers.level2.num_samples': 100, 'solver.solvers.level3.num_samples': 100})
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'fourroom_l3', overrides=ov)
    L = model.num_levels
    top = solver.level_models[L - 1]
    imagine = TopLevelImagine(top)
    cfg.world.max_episode_steps = a.steps + 10
    cfg.world.num_envs = 1
    wcfg = OmegaConf.to_container(cfg.world, resolve=True)
    wcfg.update(min_distractors=1, max_distractors=1, terminate_on_goal=False)
    world = swm.World(**wcfg, image_shape=(224, 224), verbose=0)
    world.set_policy(policy)
    env = inner_env(world)

    engine, results = None, []
    for ep in range(a.episodes):
        world.reset(seed=a.seed * 1000 + ep)
        cells = free_cells(env)
        rec = Recorder() if (a.record and ep == 0) else None
        visited, rooms, moving, prev = set(), set(), 0, None
        xy_z, goal_xy, seg_log, t0, traj = [], None, [], time.time(), []
        for t in range(a.steps):
            infos = world.infos
            xy = np.asarray(env.agent_position, dtype=np.float32).copy()     # true position, never normalised
            if t % 5 == 0:
                traj.append(xy.round(1).tolist())
            visited.add(cell_of(xy)); rooms.add(room_of(xy))
            if prev is not None:
                moving += float(np.linalg.norm(xy - prev) > 0.5)
            prev = xy.copy()
            raw_px = np.asarray(infos['pixels']).reshape(-1, 224, 224, 3)[-1].copy()

            if kind == 'random_actions':
                world.states, world.rewards, world.terminateds, world.truncateds, world.infos = \
                    world.envs.step(world.envs.action_space.sample())
                if rec: rec.frame(raw_px, xy, None, f'random t={t} cells={len(visited & cells)}')
                continue

            def encode():
                info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos.items()}
                with torch.no_grad():
                    return solver._encode_all_levels(policy._prepare_info(info), key='pixels')[f'embed_{L}'].reshape(-1)

            if engine is not None and t % 20 == 0 and t % a.segment != 0:
                engine.observe(encode())
            if t % a.segment == 0:
                z = encode()
                xy_z.append((z.cpu().numpy(), xy.copy()))
                if engine is None and kind in ('pattern', 'direct'):
                    engine = GoalEngine(z.numel(), make_stream(pname, seed=a.seed),
                                        EngineConfig(k=a.k if kind == 'pattern' else 1, learn_readout=False), seed=a.seed)
                    engine.init_imagination(horizon=a.imagine_h, action_dim=4, act_scale=a.act_scale)
                if engine is not None:
                    if t > 0:
                        engine.update(z); engine.ens.fit(steps=20, device=z.device)
                    else:
                        engine.reset(episode_offset=ep * 37)
                    engine.observe(z)
                    goal, s = engine.select_imagined(z, imagine, use_efe=(kind == 'pattern'))
                    sel = dict(G=float(s['G'][s['k']]), epi=float(s['epi'][s['k']]), nov=float(s['nov'][s['k']]))
                elif kind == 'random_macro':
                    goal = imagine(z, a.act_scale * torch.randn(1, a.imagine_h, 4, device=z.device))[0]; sel = {}
                else:
                    raise ValueError(a.goal_source)
                solver.goal_top = goal.reshape(1, 1, -1)
                policy.reset()
                if len(xy_z) >= 5:                    # linear probe z -> xy, for the replay marker only
                    Z = np.stack([p[0] for p in xy_z]); Y = np.stack([p[1] for p in xy_z]); Zm, Ym = Z.mean(0), Y.mean(0)
                    W = np.linalg.solve((Z - Zm).T @ (Z - Zm) + 10 * np.eye(Z.shape[1]), (Z - Zm).T @ (Y - Ym))
                    goal_xy = (goal.detach().cpu().numpy() - Zm) @ W + Ym
                seg_log.append(dict(t=t, cells=len(visited & cells), goal_dist=float((goal - z).norm()), **sel))

            infos['goal'] = infos['pixels']
            world.step()
            if rec:
                rec.frame(raw_px, xy, goal_xy, f'{a.goal_source} t={t} cells={len(visited & cells)}/{len(cells)}')

        cov = len(visited) / len(cells | visited)
        r = dict(episode=ep, coverage=cov, cells=len(visited), total_cells=len(cells | visited), rooms=len(rooms),
                 motion=moving / max(1, a.steps - 1), seconds=time.time() - t0, segments=seg_log, traj=traj)
        results.append(r)
        print(f'[{a.goal_source}] ep {ep}: coverage {100 * cov:.1f}% ({r["cells"]}/{len(cells)}), rooms {len(rooms)}/4, '
              f'motion {100 * r["motion"]:.0f}%, {r["seconds"]:.0f}s', flush=True)
        if rec:
            rec.save(out / f'replay_{tag}_seed{a.seed}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1, default=float))


if __name__ == '__main__':
    main()
