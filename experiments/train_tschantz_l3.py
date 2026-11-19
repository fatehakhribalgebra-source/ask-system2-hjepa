"""Train the official Tschantz ensemble on REAL level-3 transitions (state = whitened z3, action = TRUE macro-action,
target = next-state delta), with the official Normalizer, EnsembleModel.loss, Adam(lr 1e-3, eps 1e-8), grad clip
1000, batch 50, bootstrap batches per member (as the official Buffer does). Episode-split validation.

  python experiments/train_tschantz_l3.py --epochs 100
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
from system2.tschantz_l3 import CFG, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'tschantz_l3.pt'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = 'cuda'
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    d = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')
    with torch.no_grad():
        S = svae.zn(torch.from_numpy(d['z']).float().to(dev)).cpu().numpy()
        S1 = svae.zn(torch.from_numpy(d['z_next'][:, 0].astype(np.float32)).to(dev)).cpu().numpy()
    A = d['acts'][:, 0].astype(np.float32); D = S1 - S; EP = d['ep']
    val = np.isin(EP, rng.choice(np.unique(EP), len(np.unique(EP)) // 10, replace=False))
    ens, norm = build(S.shape[1], A.shape[1], dev)
    for s, ac, dl in zip(S[~val], A[~val], D[~val]):                    # official incremental normalizer
        norm.update(s, ac, dl)
    opt = torch.optim.Adam(ens.parameters(), lr=CFG['learning_rate'], eps=CFG['epsilon'])
    tS, tA, tD = (torch.from_numpy(x[~val]).float().to(dev) for x in (S, A, D))
    vS, vA, vD = (torch.from_numpy(x[val]).float().to(dev) for x in (S, A, D))
    E, B, n = ens.ensemble_size, CFG['batch_size'], len(tS)
    print(f'train {n} transitions, val {len(vS)} (episode split); ensemble {E} x hidden {CFG["hidden_size"]}')
    for ep in range(a.epochs):
        ens.train()
        for _ in range(n // B):
            idx = torch.randint(n, (E, B), device=dev)                       # bootstrap per member
            loss = ens.loss(tS[idx], tA[idx], tD[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ens.parameters(), CFG['grad_clip_norm']); opt.step()
        if ep % 10 == 9 or ep == a.epochs - 1:
            ens.eval()
            with torch.no_grad():
                dm, dv = ens(vS.unsqueeze(0).expand(E, -1, -1), vA.unsqueeze(0).expand(E, -1, -1))
                pred = vS + dm.mean(0)
                r2 = 1 - ((pred - (vS + vD)) ** 2).mean() / (vS + vD).var(0).mean()
                r2_copy = 1 - ((vS - (vS + vD)) ** 2).mean() / (vS + vD).var(0).mean()
            print(f'epoch {ep + 1}: train loss {loss.item():.2f} | val next-state R^2 {r2:.3f} (copy-current {r2_copy:.3f})', flush=True)
    torch.save({'state_dict': ens.state_dict(), 'normalizer': norm.__dict__, 'state_size': S.shape[1],
                'action_size': A.shape[1]}, a.out)
    print('saved', a.out)


if __name__ == '__main__':
    main()
