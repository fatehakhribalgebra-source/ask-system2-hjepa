"""Build the paper-style RGM (system2/rgm_paper.py) on H-JEPA level 3 or level 2, with offline checks.

Checks on held-out episodes (structure learned on the other 90%), start = first latent, goal = last latent:
  plans : how often inductive inference finds a plan, at which level, how long
  xy    : (analysis only, never seen by the agent) mean xy of each state; does the plan end near the goal's xy,
          and how far is the start from the goal compared with the plan's end?
The final model is fit on all episodes.

  python experiments/build_rgm_paper.py --level 3 --k 64
  python experiments/build_rgm_paper.py --level 2 --k 96
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_paper import PaperRGM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--level', type=int, default=3, choices=[2, 3])
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--n-levels', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seqs', default=str(ROOT / 'data' / 'latent_seqs_ant_l2_l3.npz'),
                    help='latent sequences from build_latent_seqs.py (one file per H-JEPA model seed)')
    ap.add_argument('--out', default=None)
    ap.add_argument('--cluster', default='kmeans', choices=['kmeans', 'emeans'],
                    help='level-1 states: Lloyd k-means (20 Nov 2026 models) or e-means (Bessa et al. 2026)')
    ap.add_argument('--em-epochs', type=int, default=25)
    ap.add_argument('--em-batch', type=int, default=1024)
    ap.add_argument('--em-alpha', type=float, default=0.1, help='e-means peak EMA step alpha0')
    a = ap.parse_args()
    ckw = dict(cluster=a.cluster) | (dict(epochs=a.em_epochs, batch=a.em_batch, alpha0=a.em_alpha)
                                     if a.cluster == 'emeans' else {})
    out = pathlib.Path(a.out or ROOT / 'data' / f'rgm_paper_z{a.level}_k{a.k}.pkl')
    d = np.load(a.seqs)
    Z, EP, XY = d[f'z{a.level}'].astype(np.float32), d[f'ep{a.level}'], d[f'xy{a.level}']
    rng = np.random.default_rng(a.seed)
    eps = np.unique(EP)
    val_eps = rng.choice(eps, len(eps) // 10, replace=False)
    va = np.isin(EP, val_eps)

    t0 = time.time()
    m = PaperRGM.fit(Z[~va], EP[~va], k=a.k, n_levels=a.n_levels, seed=a.seed, **ckw)
    print(f'z{a.level}, K={a.k}, {a.cluster}: fit in {time.time() - t0:.1f}s | ' + ' | '.join(
        f'level {i + 1}: {len(lv.items)} items, {sum(len(c) for c in lv.succ)} transitions'
        for i, lv in enumerate(m.levels)), flush=True)
    s_tr = m.labels
    Ytr, Yva = m.white(Z[~va]), m.white(Z[va])
    mse = lambda Y: float(((Y[:, None] - m.C[None]) ** 2).sum(-1).min(1).mean())
    cnt = np.bincount(s_tr, minlength=a.k)
    stay = np.mean([np.mean(s_tr[EP[~va] == e][1:] == s_tr[EP[~va] == e][:-1]) for e in np.unique(EP[~va])])
    sxy = np.stack([XY[~va][s_tr == j].mean(0) if (s_tr == j).any() else np.zeros(2) for j in range(a.k)])
    sp = np.array([np.linalg.norm(XY[~va][s_tr == j] - sxy[j], axis=1).mean() if (s_tr == j).sum() > 1 else 0.0
                   for j in range(a.k)])
    print(f'MSE train {mse(Ytr):.3f} val {mse(Yva):.3f} | state sizes min {cnt.min()} median {int(np.median(cnt))} '
          f'max {cnt.max()} | self-transition {stay:.2f}', flush=True)
    print(f'xy spread within a state (analysis only, cell = 4): mean {sp.mean():.2f} max {sp.max():.2f} | '
          f'aliased states (spread > 2) {int((sp > 2).sum())}, holding {int(cnt[sp > 2].sum())} points', flush=True)

    found, lv_used, lens, end_err, start_err, t_plan = 0, Counter(), [], [], [], []
    for e in val_eps:
        idx = np.flatnonzero(EP == e)
        i, j = idx[0], idx[-1]
        m.reset()
        goal = m.goal_states(Z[j])
        s0 = int(m.posterior(Z[i]).argmax())
        if s0 in goal:
            continue
        t1 = time.time()
        seq, lvl, depth = m.induce(s0, goal)
        t_plan.append(time.time() - t1)
        if seq is None:
            continue
        found += 1; lv_used[lvl] += 1; lens.append(len(seq))
        end_err.append(np.linalg.norm(sxy[seq[-1]] - XY[j])); start_err.append(np.linalg.norm(XY[i] - XY[j]))
    n = sum(1 for e in val_eps if int(m.posterior(Z[np.flatnonzero(EP == e)[0]]).argmax())
            not in m.goal_states(Z[np.flatnonzero(EP == e)[-1]]))
    print(f'plans found {found}/{n} | level used {dict(lv_used)} | plan length mean {np.mean(lens):.1f} latent steps '
          f'| planning time {1000 * np.mean(t_plan):.0f} ms', flush=True)
    print(f'goal distance (xy): start {np.mean(start_err):.2f} -> plan end {np.mean(end_err):.2f} '
          f'(median end {np.median(end_err):.2f})', flush=True)

    final = PaperRGM.fit(Z, EP, k=a.k, n_levels=a.n_levels, seed=a.seed, **ckw)
    with open(out, 'wb') as f:
        pickle.dump(final.state_dict(), f)
    print('saved', out)


from collections import Counter  # noqa: E402

if __name__ == '__main__':
    main()
