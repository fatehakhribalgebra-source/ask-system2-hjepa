"""Active-inference agent with H-JEPA level 3 as its environment (Visual AntMaze).

  --env latent : environment = level 3's predictor F3 (no ant); checks the EFE machinery and latent coverage
  --env real   : a3 executed in the maze by H-JEPA levels 2/1 (level 3 does not plan); 1 agent step = 20 env steps
Agent: state VAE perception (pretrained on real z3), transition ensemble learned online from scratch, CEM over
3-step policies scored by EFE (risk + ambiguity - parameter information gain), first action executed.
Metric (real): % maze cells visited, curve, motion, uprightness, flips. One World per process.

  python experiments/antmaze_aif.py --env real --episodes 5 --steps 1000
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

from system2.active_inference import ActiveInferenceAgent, AIFConfig, StateVAE, TransitionEnsemble  # noqa: E402
from system2.hjepa_bridge import build_policy  # noqa: E402
from system2.latent_dynamics import MacroActionTracker  # noqa: E402
from system2.level3_env import LatentActionSolver, LatentLevel3Env, RealLevel3Env  # noqa: E402

CKPT = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed{seed}' / 'ant_hjepa_l3_object.ckpt'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='real', choices=['real', 'latent'])
    ap.add_argument('--episodes', type=int, default=5)
    ap.add_argument('--steps', type=int, default=1000, help='env steps per episode (real); agent steps = steps/20')
    ap.add_argument('--w-risk', type=float, default=1.0); ap.add_argument('--w-amb', type=float, default=1.0)
    ap.add_argument('--w-ig', type=float, default=1.0)
    ap.add_argument('--random', action='store_true', help='control: uniform-random macro-actions (same scale)')
    ap.add_argument('--mode', default='eq', choices=['eq', 'legacy'],
                    help="eq = G = -state IG - E ln T(o) (- w_param * param IG); legacy = risk + amb - param IG")
    ap.add_argument('--target', default='fractal',
                    help="north star T(o): a pattern (fractal, fractal_shuffled, noise_labels, rule30, ...), "
                         "'typical' (T = N(0, I)) or 'none' (no pragmatic term)")
    ap.add_argument('--w-sig', type=float, default=1.0); ap.add_argument('--w-prag', type=float, default=1.0)
    ap.add_argument('--w-param', type=float, default=0.0); ap.add_argument('--sigma-t', type=float, default=0.5)
    ap.add_argument('--w-habit', type=float, default=1.0)
    ap.add_argument('--proto-scale', type=float, default=1.5, help='north-star move size, in std of real macro-actions')
    ap.add_argument('--tag', default='')
    ap.add_argument('--road', action='store_true', help='imagine roads in level 3 (F3), H-JEPA follows the lowest-EFE road')
    ap.add_argument('--n-roads', type=int, default=8); ap.add_argument('--road-len', type=int, default=12)
    ap.add_argument('--wp-every', type=int, default=3); ap.add_argument('--road-temp', type=float, default=1.0)
    ap.add_argument('--wp-steps', type=int, default=100, help='max env steps per waypoint')
    ap.add_argument('--off-manifold', action='store_true', help='old behaviour: raw Gaussian actions (not mapped through the LayerNorm)')
    ap.add_argument('--commit', action='store_true',
                    help='commit to the whole H-step plan: the environment steers H-JEPA toward the plan end state for H*20 env steps')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--model-seed', type=int, default=42)
    ap.add_argument('--record', action='store_true')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'antmaze_aif'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = ('random' if a.random else (f'aif_{a.target}' if a.mode == 'eq' else 'aif')) + ('_commit' if a.commit else '') + ('_road' if a.road else '') + ('_offman' if a.off_manifold else '') + a.tag + ('_latent' if a.env == 'latent' else '')
    dev = 'cuda'

    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    vk = torch.load(ROOT / 'data' / 'action_vae_ant_l3.pt', map_location=dev, weights_only=False)
    a_std = vk['state_dict']['a_std'].reshape(vk['H'], vk['A']).mean(0)          # true macro-action spread

    ov = {'eval.eval_budget': 160 if (a.commit or a.road) else 40, 'solver.solvers.level3.num_samples': 100,
          **{f'solver.solvers.level{l}.n_steps': 40 for l in (1, 2, 3)}}
    policy, solver, cfg, model = build_policy(str(CKPT).format(seed=a.model_seed), 'ant_l3', overrides=ov)
    solver = LatentActionSolver.from_solver(solver)
    policy.solver = solver
    top = solver.level_models[model.num_levels - 1]
    # valid macro-actions = outputs of level 3's action encoder, which ends in LayerNorm(4): map u -> LN(u)
    action_ln = list(top.action_encoder.children())[-1]
    assert isinstance(action_ln, torch.nn.LayerNorm), type(action_ln)
    def action_map(u):
        with torch.no_grad():
            return action_ln(u)
    ap_on = not a.off_manifold

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

    trans = TransitionEnsemble(svae.d_s, len(a_std)).to(dev)                      # learned online, from scratch
    agent = ActiveInferenceAgent(svae, trans, a_std, AIFConfig(
        w_risk=a.w_risk, w_amb=a.w_amb, w_ig=a.w_ig, mode=a.mode, w_sig=a.w_sig, w_prag=a.w_prag, w_param=a.w_param,
        sigma_t=a.sigma_t, w_habit=a.w_habit), seed=a.seed, action_map=action_map if ap_on else None)
    # ---- north star (Free Lunch): each agent action consumes one pattern label; label -> direction prototype in
    # level-3 action space (+/- first two principal axes of REAL macro-actions); target = where level 3 predicts the
    # pattern's suggested moves lead from the current state (whitened observation units).
    star = None
    if a.target not in ('typical', 'none') and a.mode == 'eq':
        from system2.patterns import make_stream
        acts_real = torch.from_numpy(np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')['acts'].reshape(-1, 4)).float()
        ev, V = torch.linalg.eigh(torch.cov((acts_real - acts_real.mean(0)).T))
        v1, v2 = V[:, -1] * ev[-1].sqrt(), V[:, -2] * ev[-2].sqrt()
        # Free Lunch order [N, E, S, W]: neighbouring labels are 90 degrees apart (0/1 alternation = zigzag, not shuttle)
        protos = (a.proto_scale * torch.stack([v1, v2, -v1, -v2]) + acts_real.mean(0)).to(dev)
        if ap_on:
            protos = action_map(protos)                                         # onto the valid manifold
        star = make_stream(a.target, seed=a.seed)
        assert star.labels is not None, 'north-star pattern must be a label image (fractal, noise_labels, rule30, ...)'

    def north_star(z, pos, H):
        labels = [int(star.labels[(pos + h) % star.length]) % 4 for h in range(H)]
        zz, ys = z.reshape(1, 1, -1), []
        with torch.no_grad():
            for lab in labels:
                zz = top.predict(zz, top.action_embed(protos[lab].reshape(1, 1, -1)))[:, -1:]
                ys.append(svae.zn(zz.reshape(1, -1))[0])
        return torch.stack(ys), labels
    H = agent.cfg.horizon
    K = a.steps // (20 * H) if a.commit else a.steps // 20

    def set_step_target(zc, p):
        if a.mode != 'eq':
            return None
        if star is not None:
            tgt, labels = north_star(zc, p, H); agent.set_target(tgt); return labels
        agent.set_target('typical' if a.target == 'typical' else None)
        return None

    def f3(zc, a3):
        with torch.no_grad():
            return top.predict(zc.reshape(1, 1, -1), top.action_embed(a3.reshape(1, 1, -1)))[:, -1].reshape(-1)

    def run_road_episode(env, z, pos, tag, log, s_visited, losses):
        """Imagine roads in level 3, score each by EFE, H-JEPA follows the lowest-EFE road; repeat until budget."""
        while env.t < a.steps:
            roads = []
            for r in range(a.n_roads):
                zi, acts, zs = z.clone(), [], []
                for n in range(a.road_len):
                    set_step_target(zi, pos + n)
                    if a.random:
                        a3 = action_map(torch.randn(len(a_std), device=dev))
                    elif r == 0:
                        a3, _ = agent.act(zi)
                    else:
                        a3, _ = agent.act_sample(zi, a.road_temp)
                    zi = f3(zi, a3); acts.append(a3); zs.append(zi)
                acts = torch.stack(acts)
                if a.mode == 'eq':
                    if star is not None:
                        agent.set_target(north_star(z, pos, a.road_len)[0])
                    else:
                        agent.set_target('typical' if a.target == 'typical' else None)
                G, terms = agent.road_efe(z, acts)
                roads.append((G, acts, zs, terms))
            best = 0 if a.random else int(np.argmin([rd[0] for rd in roads]))
            G, acts, zs, terms = roads[best]
            wps = [zs[i] for i in range(a.wp_every - 1, a.road_len, a.wp_every)]
            labels = [int(star.labels[(pos + h) % star.length]) % 4 for h in range(a.road_len)] if star is not None else None
            z2, einfo = env.follow_road(wps, max_steps_per_wp=a.wp_steps, budget_left=a.steps - env.t, label=tag)
            if not a.random:
                for zf, at, zt in einfo['realised']:
                    lv = agent.learn(zt, a_true=at, z_from=zf)
                    if lv is not None:
                        losses.append(lv)
            s_visited.append(agent.belief(z2)[0][0].cpu().numpy())
            Gs = [rd[0] for rd in roads]
            log.append(dict(t=env.t, labels=labels, road_G=G, road_G_spread=float(np.std(Gs)), **terms,
                            waypoints=einfo['waypoints'], reached=sum(w['reached'] for w in einfo['waypoints'])))
            pos += a.road_len
            z = z2
        return pos
    results = []
    for ep in range(a.episodes):
        t0 = time.time()
        rec = Recorder(loco) if (a.record and a.env == 'real') else None          # every episode
        env = RealLevel3Env(world, policy, solver, model, loco, cells, k_env=20, recorder=rec, tracker=tracker)
        z = env.reset(seed=a.seed * 1000 + ep)
        lat = LatentLevel3Env(top, z) if a.env == 'latent' else None
        log, s_visited, losses = [], [], []
        pos = ep * 37
        if a.road:
            pos = run_road_episode(env, z, pos, tag, log, s_visited, losses)
            K = 0
        for k in range(K):
            labels = None
            if a.mode == 'eq':
                if star is not None:
                    tgt, labels = north_star(z, pos, H); pos += H if a.commit else 1
                    agent.set_target(tgt)
                else:
                    agent.set_target('typical' if a.target == 'typical' else None)
            if a.random:
                rnd = (lambda *sh: action_map(torch.randn(*sh, device=dev))) if ap_on else (lambda *sh: a_std * torch.randn(*sh, device=dev))
                a3 = rnd(len(a_std)); info = {}
                plan = rnd(H, len(a_std))
            else:
                a3, info = agent.act(z)
                plan = agent.last['plan']
            if a.env == 'real' and a.commit:
                z2, einfo = env.step_plan(plan, z, label=tag)
                lv = None
                if not a.random:
                    for zf, at, zt in einfo['realised']:
                        lv = agent.learn(zt, a_true=at, z_from=zf)
            elif a.env == 'real':
                z2, einfo = env.step(a3, z, label=tag)
                rz = einfo['realised']
                lv = agent.learn(z2, a_true=None if rz is None else rz[1], z_from=None if rz is None else rz[0]) \
                    if not a.random else None
            else:
                z2, _ = lat.step(a3)
                lv = agent.learn(z2) if not a.random else None
            if lv is not None:
                losses.append(lv)
            s_visited.append(agent.belief(z2)[0][0].cpu().numpy())
            log.append(dict(k=k, labels=labels, **info, pred_err=float((einfo['predicted'] - z2).norm() / z2.norm()) if a.env == 'real' else None))
            z = z2
        if a.road:
            K = a.steps // 20
        S = np.stack(s_visited) if s_visited else np.zeros((1, svae.d_s))
        r = dict(episode=ep, seconds=time.time() - t0, agent_steps=K, model_loss=float(np.mean(losses[-5:])) if losses else None,
                 belief_spread=float(S.std(0).mean()), log=log)
        if a.env == 'real':
            r |= env.stats(a.steps)
        results.append(r)
        msg = f'[{tag}] ep {ep}: '
        if a.env == 'real':
            msg += (f'coverage {100 * r["coverage"]:.1f}% ({r["cells"]}/{r["total_cells"]}), motion {100 * r["motion"]:.0f}%, '
                    f'upright {100 * r["upright"]:.0f}% (flip at {r["first_flip"]}), ')
        ig = [l.get('ig_spread') for l in log if l.get('ig_spread') is not None]
        msg += (f'belief spread {r["belief_spread"]:.2f}, model loss {r["model_loss"]}, '
                f'IG spread {np.mean(ig) if ig else float("nan"):.2f}, {r["seconds"]:.0f}s')
        print(msg, flush=True)
        if rec is not None:
            rec.frames = rec.frames[::2]
            rec.save(out / f'replay_{tag}_seed{a.seed}_ep{ep}.mp4')
    world.close()
    (out / f'{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), episodes=results), indent=1, default=float))


if __name__ == '__main__':
    main()
