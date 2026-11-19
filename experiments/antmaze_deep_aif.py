"""Phase 2: the deep active-inference agent (trained on the z3 game) explores the real AntMaze.

Loop: encode the real z3 -> MCTS on EFE (more simulations than in training) -> the principal variation (most visited
path of moves) is the road; its F3-predicted states, every `wp_every` moves, are waypoints -> the frozen H-JEPA
hierarchy follows them (goal_top = waypoint) -> exact realised level-3 transitions (move = nearest prototype) keep
training the agent -> replan from the real z3. Control: --random-road (random moves, same following).
Metric: % maze cells (true xy), curve, motion, uprightness, flips; one replay per episode (camera | whole maze).

  python experiments/antmaze_deep_aif.py --agent results/deep_aif_z3/agent_fractal_wp1_seed0.pt --target fractal
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))

from system2.active_inference import StateVAE  # noqa: E402
from system2.deep_aif import DeepAIFAgent, DeepAIFConfig  # noqa: E402
from system2.hjepa_bridge import build_policy  # noqa: E402
from system2.latent_dynamics import MacroActionTracker  # noqa: E402
from system2.level3_env import LatentActionSolver, RealLevel3Env  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed{seed}' / 'ant_hjepa_l3_object.ckpt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--agent', default='', help='phase-1 checkpoint; empty = untrained agent')
    ap.add_argument('--target', default='fractal')
    ap.add_argument('--random-road', action='store_true')
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--sims', type=int, default=128); ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--wp-every', type=int, default=2); ap.add_argument('--wp-steps', type=int, default=100)
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze_deep_aif'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = ('random_road' if a.random_road else f'deepaif_{a.target}' + ('' if a.agent else '_untrained')) + a.tag
    dev = 'cuda'
    assets = torch.load(ROOT / 'data' / 'z3_game_assets.pt', map_location=dev, weights_only=False)
    protos = assets['protos'].to(dev)
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    agent = DeepAIFAgent(svae, protos, DeepAIFConfig(sims=a.sims, depth=a.depth), seed=a.seed)
    if a.agent:
        agent.load_state_dict(torch.load(a.agent, map_location=dev, weights_only=False))

    ov = {'eval.eval_budget': 160, 'solver.solvers.level3.num_samples': 100,
          **{f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)}}
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'ant_l3', overrides=ov)
    solver = LatentActionSolver.from_solver(solver); policy.solver = solver
    top = solver.level_models[model.num_levels - 1]

    @torch.no_grad()
    def f3(z, k):
        return top.predict(z.reshape(1, 1, -1), top.action_embed(protos[k].reshape(1, 1, -1)))[:, -1].reshape(-1)

    star = None
    if a.target not in ('typical', 'none'):
        from system2.patterns import make_stream
        star = make_stream(a.target, seed=a.seed)
        P = protos.cpu().numpy(); order = [0]
        while len(order) < 4:
            dmin = np.min([np.linalg.norm(P - P[j], axis=1) for j in order], axis=0); dmin[order] = -1
            order.append(int(np.argmax(dmin)))
        star_moves = order

    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from stable_worldmodel.world import _find_env_method
    from antmaze_explore import Recorder
    wcfg = OmegaConf.to_container(cfg.world, resolve=True)
    wcfg.update(num_envs=1, max_episode_steps=a.steps + 30, terminate_at_goal=False, dataset_type='explore')
    world = swm.World(**wcfg, image_shape=(64, 64), verbose=0)
    world.set_policy(policy)
    loco = _find_env_method(world.envs.unwrapped.envs[0], 'neutral_info_from_xy').__self__
    cells = {tuple(c) for c in loco.all_cells}
    nstats = torch.load(pathlib.Path(str(CKPT).format(seed=a.model_seed)).parent / 'normalizer.pt', weights_only=False)['stats']
    tracker = MacroActionTracker(model, nstats)

    results = []
    for ep in range(a.episodes):
        t0 = time.time()
        rec = Recorder(loco) if a.record else None
        env = RealLevel3Env(world, policy, solver, model, loco, cells, k_env=20, recorder=rec, tracker=tracker)
        z = env.reset(seed=a.seed * 1000 + ep)
        pos, log = ep * 37, []
        while env.t < a.steps:
            if a.random_road:
                pv = [int(rng.integers(len(protos))) for _ in range(a.depth)]; info = {}
            else:
                if star is not None:
                    def tgt(depth, z=z, pos=pos):
                        zz = z
                        for h in range(depth + 1):
                            zz = f3(zz, star_moves[int(star.labels[(pos + h) % star.length]) % 4])
                        return svae.zn(zz.reshape(1, -1))[0]
                    agent.target = tgt
                else:
                    agent.target = 'typical' if a.target == 'typical' else None
                pi, pv, info = agent.search(z)
                if not pv:
                    pv = [int(torch.argmax(pi))]
            zs, zz = [], z
            for k in pv:
                zz = f3(zz, k); zs.append(zz)
            wps = [zs[i] for i in range(a.wp_every - 1, len(zs), a.wp_every)] or [zs[-1]]
            z2, einfo = env.follow_road(wps, max_steps_per_wp=a.wp_steps, budget_left=a.steps - env.t, label=tag)
            if not a.random_road:
                for zf, at, zt in einfo['realised']:
                    kk = int(((protos - at.reshape(1, -1)) ** 2).sum(-1).argmin())
                    agent.remember(agent.belief(zf), kk, agent.belief(zt))
                if einfo['realised']:
                    agent.fit(steps=10)
            log.append(dict(t=env.t, road=pv, waypoints=einfo['waypoints'],
                            reached=sum(w['reached'] for w in einfo['waypoints'])))
            pos += len(pv); z = z2
        r = dict(episode=ep, seconds=time.time() - t0, roads=len(log), log=log) | env.stats(a.steps)
        results.append(r)
        wp_total = sum(len(l['waypoints']) for l in log); wp_ok = sum(l['reached'] for l in log)
        print(f'[{tag}] ep {ep}: coverage {100 * r["coverage"]:.1f}% ({r["cells"]}/{r["total_cells"]}), roads {len(log)}, '
              f'waypoints reached {wp_ok}/{wp_total}, motion {100 * r["motion"]:.0f}%, upright {100 * r["upright"]:.0f}% '
              f'(flip at {r["first_flip"]}), {r["seconds"]:.0f}s', flush=True)
        if rec is not None:
            rec.frames = rec.frames[::2]
            rec.save(out / f'replay_{tag}_seed{a.seed}_ep{ep}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1, default=float))


if __name__ == '__main__':
    main()
