"""Reward-free exploration in Crafter (Craftax-Classic) with a frozen H-JEPA trained on CrafterPlay.

Conditions (--goal-source), same as AntMaze:
  pattern:<name>   System 2 (pattern -> reservoir -> K imagined top-level goals -> EFE -> H-JEPA planner)
  direct:<name>    pattern -> reservoir -> 1 imagined goal, no EFE selection (Free-Lunch-style mapping)
  random_macro     random abstract macro-actions -> imagined goal -> H-JEPA planner
  fl_direct:<name> Free Lunch literal: pattern labels -> LEFT/RIGHT/UP/DOWN primitive actions (no model)
  random_actions   uniform random over the 17 actions
Metrics: achievements unlocked (22), Crafter score over episodes, % of world tiles visited, % time in motion.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import gymnasium as gym
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))

from envs.crafter_actions import decode  # noqa: E402
from envs.craftax_env import ACHIEVEMENTS, Craftax, crafter_score  # noqa: E402
from system2.goal_engine import EngineConfig, GoalEngine  # noqa: E402
from system2.hjepa_bridge import TopLevelImagine, build_policy  # noqa: E402
from system2.patterns import make_stream  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'crafter' / 'crafter_ft_hjepa_l3' / 'seed42' / 'crafter_ft_hjepa_l3_object.ckpt'
IMG = 224   # FourRoom-initialised model input size
PALETTE = np.array([[0, 0, 0], [80, 80, 80], [120, 180, 70], [40, 90, 220], [110, 110, 110], [30, 120, 30],
                    [130, 90, 50], [60, 60, 60], [170, 140, 120], [100, 220, 230], [140, 90, 40], [200, 60, 20],
                    [200, 200, 80], [90, 160, 60], [90, 160, 60], [90, 160, 60], [90, 160, 60]], np.uint8)


class FakeVecEnv:
    num_envs = 1
    action_space = gym.spaces.Box(-3, 3, shape=(1, 14), dtype=np.float32)


class Recorder:
    def __init__(self):
        self.frames, self.traj = [], []

    def frame(self, o, goal_txt, n_ach, t, src):
        import cv2
        view = cv2.resize(o['frame144'], (288, 288), interpolation=cv2.INTER_NEAREST)
        m = PALETTE[np.clip(o['map'], 0, len(PALETTE) - 1)]
        self.traj.append(tuple(o['pos']))
        for (x, y) in self.traj[-2000:]:
            m[x, y] = (255, 0, 255)
        m[o['pos'][0], o['pos'][1]] = (255, 255, 255)
        m = cv2.resize(m.transpose(1, 0, 2).copy(), (288, 288), interpolation=cv2.INTER_NEAREST)
        out = np.concatenate([view, m], 1)
        cv2.putText(out, f'{src} t={t} achievements={n_ach}', (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        if goal_txt:
            cv2.putText(out, goal_txt, (6, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        self.frames.append(out)

    def save(self, path):
        import imageio
        imageio.mimsave(path, self.frames, fps=15, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-source', default='pattern:fractal')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--segment', type=int, default=60)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--imagine-h', type=int, default=3)
    ap.add_argument('--act-scale', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'crafter'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = a.goal_source.replace(':', '_')
    kind, _, pname = a.goal_source.partition(':')
    uses_model = kind in ('pattern', 'direct', 'random_macro')

    if uses_model:
        policy, solver, cfg, model = build_policy(a.ckpt, 'crafter_l3', overrides={'eval.eval_budget': a.segment})
        policy.set_env(FakeVecEnv())
        L = model.num_levels
        top = solver.level_models[L - 1]
        imagine = TopLevelImagine(top)

    env = Craftax(seed=a.seed, size=IMG)
    engine, results = None, []
    unlock_counts = np.zeros(len(ACHIEVEMENTS))
    for ep in range(a.episodes):
        o = env.reset()
        rec = Recorder() if (a.record and ep == 0) else None
        visited, moving, prev = set(), 0, None
        stream = make_stream(pname, seed=a.seed + ep) if kind in ('fl_direct',) else None
        if stream is not None:
            stream.reset(0)
        t0, steps_done, goal_txt, seg_log = time.time(), 0, '', []
        for t in range(a.steps):
            visited.add(tuple(o['pos']))
            if prev is not None:
                moving += float(np.any(o['pos'] != prev))
            prev = o['pos'].copy()

            if kind == 'random_actions':
                act = np.random.randint(17)
            elif kind == 'fl_direct':
                lab = int(np.argmax(stream.next()[:4])) if stream.labels is not None else np.random.randint(4)
                act = [1, 2, 3, 4][lab % 4]              # label -> LEFT/RIGHT/UP/DOWN
            else:
                info = {'pixels': o['pixels'][None, None], 'goal': o['pixels'][None, None]}
                if t % a.segment == 0:
                    prep = policy._prepare_info({k: v.copy() for k, v in info.items()})
                    with torch.no_grad():
                        z = solver._encode_all_levels(prep, key='pixels')[f'embed_{L}'].reshape(-1)
                    if engine is None and kind in ('pattern', 'direct'):
                        engine = GoalEngine(z.numel(), make_stream(pname, seed=a.seed),
                                            EngineConfig(k=a.k if kind == 'pattern' else 1, learn_readout=False),
                                            seed=a.seed)
                        engine.init_imagination(horizon=a.imagine_h, action_dim=4, act_scale=a.act_scale)
                    if engine is not None:
                        if t > 0:
                            engine.update(z); engine.ens.fit(steps=20, device=z.device)
                        else:
                            engine.reset(episode_offset=ep * 37)
                        engine.observe(z)
                        goal, s = engine.select_imagined(z, imagine, use_efe=(kind == 'pattern'))
                        goal_txt = f'G={float(s["G"][s["k"]]):+.2f} epi={float(s["epi"][s["k"]]):.2f} nov={float(s["nov"][s["k"]]):.1f}'
                    else:
                        acts = a.act_scale * torch.randn(1, a.imagine_h, 4, device=z.device)
                        goal = imagine(z, acts)[0]
                    solver.goal_top = goal.reshape(1, 1, -1)
                    policy.reset()
                    seg_log.append(dict(t=t, tiles=len(visited), ach=int(o['achievements'].sum()),
                                        goal_dist=float((goal - z).norm())))
                elif engine is not None and t % 18 == 0:
                    prep = policy._prepare_info({k: v.copy() for k, v in info.items()})
                    with torch.no_grad():
                        engine.observe(solver._encode_all_levels(prep, key='pixels')[f'embed_{L}'].reshape(-1))
                vec = policy.get_action(info)
                act = int(decode(np.asarray(vec).reshape(-1, 14))[0])
            o, r, done = env.step(act)
            steps_done += 1
            if rec:
                rec.frame(o, goal_txt, int(o['achievements'].sum()), t, a.goal_source)
            if done:
                break
        ach = o['achievements'].astype(bool)
        unlock_counts += ach
        r = dict(episode=ep, achievements={n: bool(v) for n, v in zip(ACHIEVEMENTS, ach)}, n_achievements=int(ach.sum()),
                 tiles=len(visited), motion=moving / max(1, steps_done - 1), steps=steps_done,
                 died=steps_done < a.steps, seconds=time.time() - t0, segments=seg_log)
        results.append(r)
        print(f'[{a.goal_source}] ep {ep}: achievements {r["n_achievements"]} '
              f'({", ".join(n for n, v in r["achievements"].items() if v)}), tiles {r["tiles"]}, '
              f'motion {100 * r["motion"]:.0f}%, steps {steps_done}, {r["seconds"]:.0f}s', flush=True)
        if rec:
            rec.save(out / f'replay_{tag}_seed{a.seed}.mp4')
    rates = 100 * unlock_counts / a.episodes
    summary = dict(score=crafter_score(rates), success_rates=dict(zip(ACHIEVEMENTS, rates.tolist())),
                   mean_achievements=float(np.mean([r['n_achievements'] for r in results])),
                   mean_tiles=float(np.mean([r['tiles'] for r in results])))
    print(f'[{a.goal_source}] Crafter score {summary["score"]:.2f}%, mean achievements {summary["mean_achievements"]:.2f}, '
          f'mean tiles {summary["mean_tiles"]:.0f}', flush=True)
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), summary=summary, episodes=results),
                                                              indent=1, default=float))


if __name__ == '__main__':
    main()
