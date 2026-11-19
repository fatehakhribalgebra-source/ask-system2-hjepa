"""Train the conditional macro-action VAE on (z_top, true macro-action sequence) pairs. Split by episode.

  python experiments/train_action_vae.py --data data/macro_actions_ant_l3_h5.npz --out data/action_vae_ant_l3.pt
Reports: validation reconstruction error vs. the 'predict the mean' baseline, KL, and whether prior samples
match the data's per-dimension spread (they must, or System 2 imagines off-distribution actions).
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.action_vae import ActionVAE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz'))
    ap.add_argument('--out', default=str(ROOT / 'data' / 'action_vae_ant_l3.pt'))
    ap.add_argument('--d-u', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    d = np.load(a.data)
    Z, A, EP = torch.from_numpy(d['z']).float(), torch.from_numpy(d['acts']).float(), d['ep']
    eps = np.unique(EP); rng = np.random.default_rng(a.seed); val_eps = set(rng.choice(eps, len(eps) // 10, replace=False))
    va = torch.from_numpy(np.isin(EP, list(val_eps)))
    dev = 'cuda'
    Ztr, Atr, Zva, Ava = Z[~va].to(dev), A[~va].to(dev), Z[va].to(dev), A[va].to(dev)
    H, Ad = A.shape[1], A.shape[2]
    m = ActionVAE(Z.shape[1], H, Ad, a.d_u).to(dev)
    m.set_stats(Ztr, Atr)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(Ztr)
    print(f'train {n} pairs, val {len(Zva)} pairs (episode split), H={H}, A={Ad}, d_u={a.d_u}')
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        beta = a.beta * min(1.0, (ep + 1) / 10)                      # KL warm-up
        for i in range(0, n, 256):
            b = perm[i:i + 256]
            rec, kl = m(Ztr[b], Atr[b])
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == a.epochs - 1:
            m.eval()
            with torch.no_grad():
                rec, kl = m(Zva, Ava)
                base = (((Ava.reshape(len(Ava), -1) - m.a_mean) / m.a_std) ** 2).sum(-1).mean()
            print(f'epoch {ep + 1}: val recon {rec.item():.2f} (mean-baseline {base.item():.2f}, '
                  f'{100 * (1 - rec.item() / base.item()):.0f}% explained), KL {kl.item():.2f}')
    m.eval()
    with torch.no_grad():
        s = m.decode(Zva[:2000], torch.randn(min(2000, len(Zva)), a.d_u, device=dev))
    print('per-dim std  data :', Ava.reshape(-1, Ad).std(0).cpu().numpy().round(3))
    print('per-dim std  prior samples:', s.reshape(-1, Ad).std(0).cpu().numpy().round(3))
    torch.save({'state_dict': m.state_dict(), 'd_z': Z.shape[1], 'H': H, 'A': Ad, 'd_u': a.d_u}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
