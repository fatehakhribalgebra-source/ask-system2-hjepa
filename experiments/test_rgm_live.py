"""Stream test of system2/rgm_live.LiveRGM (no simulator): recorded latent sequences stand in for live play, fed one
episode after another into ONE empty model (continual), with live F3 dreaming on the GPU. The recordings are only a
test stream here -- the model never sees them in advance.

Reports, per block of episodes: states so far, and for the NEXT episodes (before they are played) how often a route
from their first latent to their last exists.

  python experiments/test_rgm_live.py --seed 42 --episodes 200 --rho 1.0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))
sys.path.insert(0, str(ROOT))
from system2.rgm_live import LiveRGM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--rho', type=float, default=1.0)
    ap.add_argument('--no-f3', action='store_true')
    a = ap.parse_args()
    dev = 'cuda'
    seqs = ROOT / 'data' / ('latent_seqs_ant_l2_l3.npz' if a.seed == 42 else f'latent_seqs_ant_l2_l3_seed{a.seed}.npz')
    q = np.load(seqs); Z, EP = q['z3'].astype(np.float32), q['ep3']
    model = torch.load(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / f'seed{a.seed}' / 'ant_hjepa_l3_object.ckpt',
                       map_location='cpu', weights_only=False).to(dev).eval()
    top = model.jepas[-1]
    ln = list(top.action_encoder.children())[-1]
    g = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        P = ln(torch.randn(12, ln.normalized_shape[0], generator=g).to(dev))

    def f3_latents(zn):
        with torch.no_grad():
            zz = torch.as_tensor(zn, device=dev).reshape(1, 1, -1).repeat(len(P), 1, 1)
            return top.predict(zz, top.action_embed(P[:, None]))[:, -1].float().cpu().numpy()

    m = LiveRGM()
    m.rho = a.rho
    if not a.no_f3:
        m.f3_latents = f3_latents
    eps = np.unique(EP)
    rng = np.random.default_rng(0)
    order = rng.permutation(eps)[: a.episodes]
    probe = rng.permutation(np.setdiff1d(eps, order))[:50]          # held-out start/goal pairs

    def routes():
        found = 0
        for e in probe:
            idx = np.flatnonzero(EP == e)
            j0, d0 = m.nearest(Z[idx[0]]); jg, dg = m.nearest(Z[idx[-1]])
            if j0 is None or d0 > m.radius or dg > m.radius:
                continue
            if j0 == jg or m.induce(j0, {jg}, blocked=m.blocked)[0] is not None:
                found += 1
        return found

    for k, e in enumerate(order, 1):
        idx = np.flatnonzero(EP == e)
        m.reset()
        for i in idx[:-1]:
            m.step(Z[i], Z[idx[-1]], lr=1.0)
        if k in (1, 5, 10, 25, 50, 100, 200, 400) or k == len(order):
            print(f'after {k:3d} episodes ({len(idx)} steps each): states {m.K:4d} | radius {m.radius:.2f} | '
                  f'held-out start->goal routes {routes()}/{len(probe)} | last episode {m.stats}', flush=True)


if __name__ == '__main__':
    main()
