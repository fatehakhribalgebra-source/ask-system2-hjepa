"""Offline checks of the fuller RGM (system2/rgm_full.py) on held-out episodes (structure learned on the other 90%,
same split as build_rgm_paper.py). xy is used ONLY to score, never by the model.

State decoding: replay each held-out episode's z3 sequence; for every step compare the decoded state's mean xy (from
training data) with the true xy. 'wrong place' = more than one maze cell (4 units) away -- aliasing errors.
Planning: induction from the first latent to the last; plan found, plan end distance to the goal's xy.

  python experiments/eval_rgm_offline.py --seqs data/latent_seqs_ant_l2_l3.npz [--f3 data/f3_transitions_seed42.npz]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_full import FullRGM  # noqa: E402
from system2.rgm_paper import PaperRGM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', default=str(ROOT / 'data' / 'latent_seqs_ant_l2_l3.npz'))
    ap.add_argument('--f3', default='', help='imagined transitions npz from build_f3_transitions.py')
    ap.add_argument('--f3-weights', type=float, nargs='*', default=[0.1, 1.0])
    ap.add_argument('--precs', type=float, nargs='*', default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    d = np.load(a.seqs)
    Z, EP, XY = d['z3'].astype(np.float32), d['ep3'], d['xy3']
    rng = np.random.default_rng(a.seed)
    eps = np.unique(EP)
    val_eps = rng.choice(eps, len(eps) // 10, replace=False)
    va = np.isin(EP, val_eps)
    base = PaperRGM.fit(Z[~va], EP[~va], k=64, n_levels=3, seed=a.seed)
    sxy = np.stack([XY[~va][base.labels == j].mean(0) for j in range(base.K)])
    state = base.state_dict()

    def decode(m, filtering, prec=1.0):
        m.filtering, m.lik_prec = filtering, prec
        err = []
        for e in val_eps:
            idx = np.flatnonzero(EP == e)
            m.reset()
            for i in idx:
                s = m.infer_state(Z[i], True)
                err.append(np.linalg.norm(sxy[s] - XY[i]))
        err = np.asarray(err)
        return err.mean(), (err > 4).mean()

    m = FullRGM.from_paper(state)
    e0, w0 = decode(m, False)
    print(f'decode alone (paper RGM):         mean xy error {e0:.2f} | wrong place {100 * w0:.1f}% of steps')
    for p in a.precs:
        e1, w1 = decode(m, True, p)
        print(f'decode filtered, lik prec {p:<4}:   mean xy error {e1:.2f} | wrong place {100 * w1:.1f}% of steps')

    def plans(m):
        found, end = 0, []
        n = 0
        for e in val_eps:
            idx = np.flatnonzero(EP == e)
            i, j = idx[0], idx[-1]
            m.reset()
            goal = m.goal_states(Z[j]); s0 = int(m.posterior(Z[i]).argmax())
            if s0 in goal:
                continue
            n += 1
            seq, lvl, _ = m.induce(s0, goal)
            if seq is not None:
                found += 1; end.append(np.linalg.norm(sxy[seq[-1]] - XY[j]))
        return found, n, float(np.mean(end))

    f, n, e = plans(FullRGM.from_paper(state))
    print(f'plans, real transitions only:       {f}/{n} found | plan end {e:.2f} from goal xy')
    if a.f3:
        q = np.load(a.f3)
        for w in a.f3_weights:
            m = FullRGM.from_paper(state)
            m.add_transitions(q['N_imag'], w)
            f, n, e = plans(m)
            e1, w1 = decode(m, True, 1.0)
            print(f'+ F3 transitions (weight {w:<4}):    {f}/{n} found | plan end {e:.2f} | '
                  f'filtered decode wrong place {100 * w1:.1f}%')


if __name__ == '__main__':
    main()
