"""Phase 1: the deep active-inference agent plays the z3 game (dynamics = H-JEPA's F3) and learns.

Each episode starts from a real z3 of the dataset; every move: MCTS on EFE -> visit distribution pi -> move sampled
from pi -> z3' = F3(z3, prototype[move]); the agent stores (s, a, s') and pi, and trains its transition model and
habit network. Exploration in the game is measured (evaluation only) with the z3 -> xy probe: distinct maze cells
visited by the probed positions. A random-move player is run on the same starts as the reference.

  python experiments/train_deep_aif_z3.py --episodes 200 --moves 30 --target fractal
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))
from system2.active_inference import StateVAE  # noqa: E402
from system2.deep_aif import DeepAIFAgent, DeepAIFConfig  # noqa: E402

MAZE_UNIT = 4.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--moves', type=int, default=30)
    ap.add_argument('--target', default='fractal', help="north star: a label pattern, 'typical' or 'none'")
    ap.add_argument('--sims', type=int, default=48)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--w-param', type=float, default=1.0); ap.add_argument('--w-sig', type=float, default=1.0)
    ap.add_argument('--w-prag', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'deep_aif_z3'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f'{a.target}_wp{a.w_param:g}'
    dev = 'cuda'
    assets = torch.load(ROOT / 'data' / 'z3_game_assets.pt', map_location=dev, weights_only=False)
    protos = assets['protos'].to(dev)
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    model = torch.load(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed42' / 'ant_hjepa_l3_object.ckpt',
                       map_location=dev, weights_only=False).eval()
    top = model.jepas[-1]
    W, zm, ym = assets['probe_W'].to(dev), assets['probe_zm'].to(dev), assets['probe_ym'].to(dev)
    probe = lambda z: ((z - zm) @ W + ym)
    cell = lambda xy: tuple((xy / MAZE_UNIT).round().int().tolist())

    @torch.no_grad()
    def f3(z, k):
        return top.predict(z.reshape(1, 1, -1), top.action_embed(protos[k].reshape(1, 1, -1)))[:, -1].reshape(-1)

    agent = DeepAIFAgent(svae, protos, DeepAIFConfig(sims=a.sims, depth=a.depth, w_param=a.w_param, w_sig=a.w_sig,
                                                    w_prag=a.w_prag), seed=a.seed)
    star = None
    if a.target not in ('typical', 'none'):
        from system2.patterns import make_stream
        star = make_stream(a.target, seed=a.seed)
        # north star: label -> a fixed move (Free Lunch order N,E,S,W) among 4 well-separated prototypes
        P = protos.cpu().numpy(); order = [0]
        while len(order) < 4:
            dmin = np.min([np.linalg.norm(P - P[j], axis=1) for j in order], axis=0); dmin[order] = -1
            order.append(int(np.argmax(dmin)))
        star_moves = order
    starts = assets['z_start'].to(dev)
    log = []
    t_all = time.time()
    for ep in range(a.episodes):
        z0 = starts[int(rng.integers(len(starts)))]
        res = {}
        for player in ('agent', 'random'):
            z, cells, pos, n_moves = z0.clone(), set(), ep * 37, []
            cells.add(cell(probe(z)))
            for m in range(a.moves):
                if player == 'agent':
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
                    k = int(torch.distributions.Categorical(probs=pi).sample())
                    z2 = f3(z, k)
                    agent.remember(agent.belief(z), k, agent.belief(z2), pi)
                    n_moves.append(k)
                else:
                    k = int(rng.integers(len(protos))); z2 = f3(z, k)
                z, pos = z2, pos + 1
                cells.add(cell(probe(z)))
            res[player] = len(cells)
            if player == 'agent':
                fit = agent.fit(steps=30)
                moves_used = len(set(n_moves))
        log.append(dict(episode=ep, agent_cells=res['agent'], random_cells=res['random'], moves_used=moves_used, **(fit or {})))
        if ep % 10 == 9 or ep == a.episodes - 1:
            L = log[-10:]
            print(f'[{tag}] ep {ep + 1}: cells visited in imagined play (last 10 eps): agent '
                  f'{np.mean([l["agent_cells"] for l in L]):.1f} vs random {np.mean([l["random_cells"] for l in L]):.1f} | '
                  f'distinct moves used {np.mean([l["moves_used"] for l in L]):.1f}/{len(protos)} | '
                  f'trans nll {L[-1].get("trans_nll", float("nan")):.2f} habit KL {L[-1].get("habit_kl", float("nan")):.2f} | '
                  f'{time.time() - t_all:.0f}s', flush=True)
    torch.save(agent.state_dict() | {'args': vars(a)}, out / f'agent_{tag}_seed{a.seed}.pt')
    (out / f'train_{tag}_seed{a.seed}.json').write_text(json.dumps(dict(args=vars(a), log=log), indent=1))


if __name__ == '__main__':
    main()
