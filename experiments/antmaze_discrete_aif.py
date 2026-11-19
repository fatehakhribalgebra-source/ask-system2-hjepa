"""Discrete active inference on a latent place map, H-JEPA walks between places (real AntMaze).

  python experiments/antmaze_discrete_aif.py --target fractal --episodes 5 --steps 1000
Controls: --w-nov 0 (no novelty), --w-prag 0 / --target none (no north star), --target fractal_shuffled,
--random (random neighbouring place). The Dirichlet counts persist across episodes (the agent keeps its map).
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
from system2.discrete_aif import DiscreteAIF, PlaceMap  # noqa: E402
from system2.hjepa_bridge import build_policy  # noqa: E402
from system2.level3_env import LatentActionSolver, RealLevel3Env  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed{seed}' / 'ant_hjepa_l3_object.ckpt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='fractal')
    ap.add_argument('--w-nov', type=float, default=1.0); ap.add_argument('--w-prag', type=float, default=1.0)
    ap.add_argument('--places', type=int, default=64); ap.add_argument('--nb', type=int, default=6)
    ap.add_argument('--horizon', type=int, default=3)
    ap.add_argument('--random', action='store_true')
    ap.add_argument('--goal-obs', action='store_true', help='goal = nearest remembered real observation (paper goal path)')
    ap.add_argument('--episodes', type=int, default=5); ap.add_argument('--steps', type=int, default=1000)
    ap.add_argument('--wp-steps', type=int, default=300)
    ap.add_argument('--nb-from', default='experience', choices=['experience', 'latent'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze_discrete_aif'))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = ('random_place' if a.random else f'daif_{a.target}_nov{a.w_nov:g}_prag{a.w_prag:g}') + ('_obs' if a.goal_obs else '') + a.tag
    dev = 'cuda'
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    Z = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')['z']
    with torch.no_grad():
        Y = svae.zn(torch.from_numpy(Z).float().to(dev)).cpu().numpy()
    pmap = PlaceMap(Y, Z, n_places=a.places, n_nb=a.nb, seed=a.seed)
    if a.nb_from == 'experience':
        Zn = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')['z_next'][:, 1].astype(np.float32)   # 2 level-3 steps on
        with torch.no_grad():
            Yn = svae.zn(torch.from_numpy(Zn).float().to(dev)).cpu().numpy()
        pmap.set_experience_neighbours(Y, Yn, a.nb)
    agent = DiscreteAIF(pmap, w_nov=a.w_nov, w_prag=a.w_prag if a.target != 'none' else 0.0, horizon=a.horizon, seed=a.seed)
    star = None
    if a.target not in ('none', 'typical'):
        from system2.patterns import make_stream
        star = make_stream(a.target, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    ov = {'eval.eval_budget': max(350, a.wp_steps + 10), 'solver.solvers.level3.num_samples': 100,
          **{f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)}}
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'ant_l3', overrides=ov)
    solver = LatentActionSolver.from_solver(solver); policy.solver = solver
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
    goals = torch.from_numpy(pmap.goal).float().to(dev)
    if a.goal_obs:
        from system2.goal_bank import GoalBank
        bank = GoalBank(ROOT / 'data' / 'datasets' / 'visual_antmaze_medium_stitch_val.h5',
                        ROOT / 'data' / 'macro_actions_ant_l3_h5.npz', svae)
        goals = [bank.nearest(torch.from_numpy(pmap.C[j]).float().to(dev), whitened=True) for j in range(pmap.n)]
    place_of = lambda z: pmap.place_of(svae.zn(z.reshape(1, -1)).detach().cpu().numpy()[0])

    results = []
    for ep in range(a.episodes):
        t0 = time.time()
        rec = Recorder(loco) if a.record else None
        env = RealLevel3Env(world, policy, solver, model, loco, cells, k_env=20, recorder=rec, tracker=None)
        z = env.reset(seed=a.seed * 1000 + ep)
        pos, log, places_seen = ep * 37, [], set()
        s = place_of(z); places_seen.add(s)
        while env.t < a.steps:
            labels = None if star is None else [int(star.labels[(pos + h) % star.length]) % 4 for h in range(a.horizon)]
            if a.random:
                i, info = int(rng.integers(pmap.nb.shape[1])), {}
            else:
                i, info = agent.choose(s, labels)
            j = int(pmap.nb[s, i])
            z2, einfo = env.follow_road([goals[j]], max_steps_per_wp=a.wp_steps, budget_left=a.steps - env.t, label=tag)
            s2 = place_of(z2)
            agent.update(s, i, s2)
            log.append(dict(t=env.t, s=s, target=j, arrived=s2, hit=s2 == j, label=None if labels is None else labels[0], **info))
            places_seen.add(s2); s, z, pos = s2, z2, pos + 1
        r = dict(episode=ep, seconds=time.time() - t0, decisions=len(log), latent_places=len(places_seen),
                 hits=sum(l['hit'] for l in log), log=log) | env.stats(a.steps)
        results.append(r)
        print(f'[{tag}] ep {ep}: coverage {100 * r["coverage"]:.1f}% ({r["cells"]}/{r["total_cells"]}), decisions {len(log)}, '
              f'target place reached {r["hits"]}/{len(log)}, latent places {len(places_seen)}, motion {100 * r["motion"]:.0f}%, '
              f'upright {100 * r["upright"]:.0f}% (flip at {r["first_flip"]}), {r["seconds"]:.0f}s', flush=True)
        if rec is not None:
            rec.frames = rec.frames[::2]
            rec.save(out / f'replay_{tag}_seed{a.seed}_ep{ep}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1, default=float))


if __name__ == '__main__':
    main()
