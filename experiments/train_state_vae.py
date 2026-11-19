"""Train the agent's perception model: state VAE q(s | z3), p(z3 | s) on real level-3 latents (episode split).

  python experiments/train_state_vae.py --d-s 16
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.active_inference import StateVAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'state_vae_ant_l3.pt'))
    ap.add_argument('--d-s', type=int, default=16)
    ap.add_argument('--d-obs', type=int, default=32, help='whitened PCA dims the agent observes (0 = all)')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(a.data)
    Z = np.concatenate([d['z'], d['z_next'][:, -1].astype(np.float32)])           # distinct-ish real latents
    EP = np.concatenate([d['ep'], d['ep']])
    rng = np.random.default_rng(a.seed); eps = np.unique(EP); val = set(rng.choice(eps, len(eps) // 10, replace=False))
    va = np.isin(EP, list(val))
    dev = 'cuda'
    Ztr, Zva = torch.from_numpy(Z[~va]).float().to(dev), torch.from_numpy(Z[va]).float().to(dev)
    m = StateVAE(Z.shape[1], a.d_s, d_obs=a.d_obs or None).to(dev)
    m.z_mean.copy_(Ztr.mean(0)); m.z_std.copy_(Ztr.std(0) + 1e-6)
    if a.d_obs:
        m.set_whitener(Ztr, a.d_obs)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f'train {len(Ztr)} latents, val {len(Zva)} (episode split), d_s={a.d_s}')
    for ep in range(a.epochs):
        perm = torch.randperm(len(Ztr), device=dev)
        beta = min(1.0, (ep + 1) / 10)
        for i in range(0, len(Ztr), 512):
            loss, _, _ = m.loss(Ztr[perm[i:i + 512]], beta)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            with torch.no_grad():
                mu, _ = m.posterior(Zva); xm, xv = m.likelihood(mu)
                r2 = 1 - ((m.zn(Zva) - xm) ** 2).mean() / m.zn(Zva).var(0).mean()
                _, nll, kl = m.loss(Zva)
            print(f'epoch {ep + 1}: val NLL {nll.item():.1f}, KL {kl.item():.1f}, reconstruction R^2 {r2.item():.3f}, '
                  f'decoder std (mean) {xv.sqrt().mean().item():.3f}')
    torch.save({'state_dict': m.state_dict(), 'd_z': Z.shape[1], 'd_s': a.d_s, 'd_obs': a.d_obs or None}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
