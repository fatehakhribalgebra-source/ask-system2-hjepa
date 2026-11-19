"""Diagnose the pretrained AntMaze H-JEPA eval on single tasks with the exact planning_eval pipeline.

Records per step: ant xy, distance to goal xy, torso height and uprightness, and saves a side-by-side
video (agent view | goal image). Usage:
  python experiments/antmaze_diag.py --tasks 1 2 --out results/antmaze_diag
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
HJ = ROOT / 'H-JEPA' / 'h_jepa'
sys.path.insert(0, str(HJ))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
os.environ['MUJOCO_GL'] = 'glfw' if os.name == 'nt' else 'egl'

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

import planning_eval as pe  # noqa: E402
import stable_worldmodel as swm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', type=int, nargs='+', default=[1])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--goal-mode', default='full', choices=['full', 'top_only'],
                    help='full = paper (goal encoded at every level); top_only = goal only as its level-3 latent, '
                         'through ExplorationSolver (the path System 2 uses)')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze_diag'))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

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
    if a.goal_mode == 'top_only':
        sys.path.insert(0, str(ROOT))
        from system2.hjepa_bridge import ExplorationSolver

        class TopOnlyGoal(ExplorationSolver):
            def _encode_all_levels(self, info_dict, key='pixels'):
                if key != 'goal':
                    return super(ExplorationSolver, self)._encode_all_levels(info_dict, key)
                full = super(ExplorationSolver, self)._encode_all_levels(info_dict, key)
                Lq = len(self.level_models)
                return {f'embed_{Lq}': full[f'embed_{Lq}']}
        solver = TopOnlyGoal.from_solver(solver)
        solver.__class__ = TopOnlyGoal
        policy.solver = solver
    gens = [(s.torch_gen, s.torch_gen.initial_seed()) for s in solver.level_solvers.values() if hasattr(s, 'torch_gen')]
    payload = torch.load(str(HJ / cfg.load_eval_trajs_path), map_location='cpu', weights_only=False)
    callables = OmegaConf.to_container(cfg.eval.callables, resolve=True)

    summary = []
    for t in a.tasks:
        for gen, seed in gens:
            gen.manual_seed(seed + t)
        chunk = {**payload, 'data': [dict(payload['data'][t])], 'episodes_idx': [t]}
        world = swm.World(**{**world_cfg, 'num_envs': 1}, image_shape=(64, 64))
        env = world.envs.unwrapped.envs[0].unwrapped
        inner = env.env if hasattr(env, 'env') else env
        log, frames = [], []
        orig_step = world.step

        def step_and_log():
            orig_step()
            q = np.asarray(inner.data.qpos).copy()
            gxy = np.asarray(world.infos['goal_xy']).reshape(-1)[:2]
            px = np.asarray(world.infos['pixels']).reshape(-1, 64, 64, 3)[-1]
            goal = np.asarray(world.infos['goal']).reshape(-1, 64, 64, 3)[-1]
            log.append(dict(x=float(q[0]), y=float(q[1]), z=float(q[2]), up=float(q[3] ** 2 + q[6] ** 2 - q[4] ** 2 - q[5] ** 2),
                            dist=float(np.linalg.norm(q[:2] - gxy)), gx=float(gxy[0]), gy=float(gxy[1])))
            frames.append(np.concatenate([px, goal], 1))

        world.step = step_and_log
        world.set_policy(policy)
        try:
            with torch.enable_grad(), sdpa_kernel(SDPBackend.MATH):
                m = world.evaluate_from_dataset(
                    None, episodes_idx=None, start_steps=None, goal_offset_steps=cfg.eval.goal_offset_steps,
                    eval_budget=cfg.eval.eval_budget, callables=callables, load_eval_trajs_path=chunk, process=process,
                    start_state_mode=cfg.eval.start_state_mode, goal_state_mode=cfg.eval.goal_state_mode)
        finally:
            world.close()
        d = [r['dist'] for r in log]
        s = dict(task=t, success=bool(m['episode_successes'][0]), steps=len(log), dist_start=d[0], dist_min=min(d),
                 dist_end=d[-1], z_min=min(r['z'] for r in log), up_min=min(r['up'] for r in log),
                 expert_len=int(m['loaded_eval_original_lengths'][0]))
        print(json.dumps(s), flush=True)
        summary.append(s)
        (out / f'task{t:02d}_{a.goal_mode}.json').write_text(json.dumps(dict(summary=s, log=log)))
        import imageio
        imageio.mimsave(out / f'task{t:02d}_{a.goal_mode}.mp4', [np.kron(f, np.ones((4, 4, 1), np.uint8)) for f in frames[::2]],
                        fps=20, macro_block_size=1)
    (out / f'summary_{a.goal_mode}_{"_".join(map(str, a.tasks))}.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
