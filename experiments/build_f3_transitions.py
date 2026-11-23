"""Imagined level-1 transitions for the RGM from H-JEPA's level-3 predictor F3 (one step = 20 env steps).

Following the 20 Nov notes: keep rollouts SHORT (one F3 step: one-step predictions are accurate, long ones drift),
start from REAL latents, use REAL macro-actions (k-means prototypes of the true level-3 actions, mapped through the
action-encoder LayerNorm so they lie on the manifold F3 was trained on), SNAP each imagined latent onto the RGM's
existing states and drop imagined latents that fit no state (out of distribution). Counts are saved separately so
the RGM can weight dreams below experience.

Checks: with the TRUE macro-action, how often does F3's snapped next state equal the real next state (and how often
does the real step change state at all); how many imagined edges are new to the real graph.

  python experiments/build_f3_transitions.py --seed 42
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))
sys.path.insert(0, str(ROOT))
from system2.rgm_full import FullRGM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--k', type=int, default=12, help='macro-action prototypes')
    ap.add_argument('--model', default='', help='RGM pickle whose states to snap onto (default: paper-style k-means)')
    ap.add_argument('--out', default='')
    ap.add_argument('--fit-quantile', type=float, default=0.95,
                    help='keep an imagined latent only if its distance to the nearest state is within this quantile '
                         'of real latents\' distances')
    a = ap.parse_args()
    dev = 'cuda'
    ck = ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / f'seed{a.seed}' / 'ant_hjepa_l3_object.ckpt'
    model = torch.load(ck, map_location='cpu', weights_only=False).to(dev).eval()
    jepa = model.jepas[-1]
    ln = list(jepa.action_encoder.children())[-1]
    d = np.load(ROOT / 'data' / f'macro_actions_ant_l3_h1_seed{a.seed}.npz')
    Z, A, ZN = d['z'].astype(np.float32), d['acts'][:, 0].astype(np.float32), d['z_next'][:, 0].astype(np.float32)
    with open(a.model or ROOT / 'data' / f'rgm_paper_z3_k64_seed{a.seed}.pkl', 'rb') as f:
        m = FullRGM.from_paper(pickle.load(f))

    def snap(X):
        Y = m.white(X)
        d2 = np.concatenate([((Y[i:i + 4096, None] - m.C[None]) ** 2).sum(-1) for i in range(0, len(Y), 4096)])
        return d2.argmin(1), d2.min(1)

    def f3(Zb, Ab):
        out = []
        with torch.no_grad():
            for i in range(0, len(Zb), 2048):
                z = torch.from_numpy(Zb[i:i + 2048]).to(dev)[:, None]
                act = torch.from_numpy(Ab[i:i + 2048]).to(dev)[:, None]
                out.append(jepa.predict(z, jepa.action_embed(act))[:, -1].float().cpu().numpy())
        return np.concatenate(out)

    s0, d_real = snap(Z)
    s1, _ = snap(ZN)
    thr = np.quantile(d_real, a.fit_quantile)
    # 1. F3 with the true macro-action
    sp, dp = snap(f3(Z, A))
    moved = s1 != s0
    print(f'seed {a.seed}: {len(Z)} real level-3 transitions, state changes in {100 * moved.mean():.0f}%')
    print(f'F3 (true action) snapped next state = real next state: {100 * (sp == s1).mean():.1f}% overall, '
          f'{100 * (sp[moved] == s1[moved]).mean():.1f}% when the state changes '
          f'(staying put would score {100 * (s0 == s1).mean():.1f}% / 0%) | fits a state {100 * (dp <= thr).mean():.0f}%')
    # 2. imagined transitions under every prototype
    rng = np.random.default_rng(0)
    C = A[rng.choice(len(A), a.k, replace=False)].copy()
    for _ in range(50):
        lab = ((A[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([A[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(a.k)])
    with torch.no_grad():
        P = ln(torch.from_numpy(C).float().to(dev)).cpu().numpy()
    K = m.K
    N_imag = np.zeros((K, K))
    kept = 0
    for p in P:
        si, di = snap(f3(Z, np.repeat(p[None], len(Z), 0)))
        ok = di <= thr
        kept += ok.sum()
        np.add.at(N_imag, (si[ok], s0[ok]), 1)
    real_edges = m.N_real > 0
    imag_edges = N_imag > 0
    new = imag_edges & ~real_edges
    print(f'imagined: {kept}/{len(Z) * a.k} latents fit a state | edges real {int(real_edges.sum())}, imagined '
          f'{int(imag_edges.sum())}, new to the real graph {int(new.sum())} (carrying {100 * N_imag[new].sum() / N_imag.sum():.1f}% '
          f'of imagined mass)')
    out = pathlib.Path(a.out) if a.out else ROOT / 'data' / f'f3_transitions_seed{a.seed}.npz'
    np.savez(out, N_imag=N_imag, protos=P, thr=thr)
    print('saved', out)


if __name__ == '__main__':
    main()
