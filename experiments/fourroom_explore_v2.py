"""FourRoom exploration with System 2 v2 (commitment, place-novelty, mixed proposals, learning readout).

Conditions (--goal-source):  v2:<pattern>   e.g. v2:fractal, v2:fractal_shuffled, v2:lorenz
                             v2:none        no pattern pathway (all candidates diverse/random) -- control
Add --no-learn to freeze the readout. Same metrics/JSON layout as fourroom_explore.py.
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

from experiments.fourroom_explore import CKPT, Recorder, cell_of, free_cells, inner_env, room_of  # noqa: E402
from system2.goal_engine_v2 import GoalEngineV2, V2Config  # noqa: E402
from system2.hjepa_bridge import TopLevelImagine, build_policy  # noqa: E402
from system2.patterns import make_stream  # noqa: E402

import stable_worldmodel as swm  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

CHECK = 20   # env steps per commitment check (= one level-3 step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal-source', default='v2:fractal')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--no-learn', action='store_true')
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'fourroom'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pname = a.goal_source.split(':', 1)[1]
    tag = a.goal_source.replace(':', '_') + ('_nolearn' if a.no_learn else '')

    ov = {'eval.eval_budget': 120, **{f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)},
          'solver.solvers.level2.num_samples': 100, 'solver.solvers.level3.num_samples': 100}   # = --fast in v1
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'fourroom_l3', overrides=ov)
    L = model.num_levels
    imagine = TopLevelImagine(solver.level_models[L - 1])
    cfg.world.max_episode_steps = a.steps + 10; cfg.world.num_envs = 1
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
        visited, rooms, moving, prev, traj, seg_log, t0 = set(), set(), 0, None, [], [], time.time()
        for t in range(a.steps):
            infos = world.infos
            xy = np.asarray(env.agent_position, dtype=np.float32).copy()
            if t % 5 == 0:
                traj.append(xy.round(1).tolist())
            visited.add(cell_of(xy)); rooms.add(room_of(xy))
            if prev is not None:
                moving += float(np.linalg.norm(xy - prev) > 0.5)
            prev = xy.copy()
            raw_px = np.asarray(infos['pixels']).reshape(-1, 224, 224, 3)[-1].copy()
            if t % CHECK == 0:
                info = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in infos.items()}
                with torch.no_grad():
                    z = solver._encode_all_levels(policy._prepare_info(info), key='pixels')[f'embed_{L}'].reshape(-1)
                if engine is None:
                    stream = None if pname == 'none' else make_stream(pname, seed=a.seed)
                    engine = GoalEngineV2(z.numel(), stream, V2Config(learn_readout=not a.no_learn,
                                                                       use_pattern=stream is not None), seed=a.seed)
                if t == 0:
                    engine.reset(episode_offset=ep * 37)
                engine.observe(z)
                if engine.should_reselect(z):
                    if engine.goal is not None:
                        engine.finish_goal(z)
                    goal, s = engine.select(z, imagine)
                    solver.goal_top = goal.reshape(1, 1, -1)
                    policy.reset()
                    seg_log.append(dict(t=t, cells=len(visited), **s))
            infos['goal'] = infos['pixels']
            world.step()
            if rec:
                rec.frame(raw_px, xy, None, f'{a.goal_source} t={t} cells={len(visited)}/{len(cells | visited)}')
        cov = len(visited) / len(cells | visited)
        r = dict(episode=ep, coverage=cov, cells=len(visited), total_cells=len(cells | visited), rooms=len(rooms),
                 motion=moving / max(1, a.steps - 1), seconds=time.time() - t0, segments=seg_log, traj=traj,
                 goals=len(seg_log), from_pattern=sum(s.get('from_pattern', False) for s in seg_log))
        results.append(r)
        print(f'[{tag}] ep {ep}: coverage {100 * cov:.1f}% ({r["cells"]}/{r["total_cells"]}), rooms {len(rooms)}/4, '
              f'goals {r["goals"]} (pattern {r["from_pattern"]}), motion {100 * r["motion"]:.0f}%, '
              f'{r["seconds"]:.0f}s', flush=True)
        if rec:
            rec.save(out / f'replay_{tag}_seed{a.seed}.mp4')
    world.close()
    args = vars(a) | {'goal_source': tag.replace('v2_', 'v2:', 1)}
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=args, episodes=results), indent=1, default=float))


if __name__ == '__main__':
    main()
