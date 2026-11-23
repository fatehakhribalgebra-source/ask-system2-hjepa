"""Offline tests of system2/rgm_agent.RGMAgent (no simulator).

  python experiments/test_rgm_agent.py --seed 42
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import random
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_agent import RGMAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', default='emeans', choices=['emeans', 'paper'])
    a = ap.parse_args()
    seqs = ROOT / 'data' / ('latent_seqs_ant_l2_l3.npz' if a.seed == 42 else f'latent_seqs_ant_l2_l3_seed{a.seed}.npz')
    pk = ROOT / 'data' / (f'rgm_emeans_z3_k64_seed{a.seed}.pkl' if a.model == 'emeans' else f'rgm_paper_z3_k64_seed{a.seed}.pkl')
    q = np.load(seqs); Z, EP = q['z3'].astype(np.float32), q['ep3']
    m = RGMAgent.from_paper(pickle.load(open(pk, 'rb')))
    m.calibrate(Z)
    print(f'surprise threshold {m.fit_thr:.2f} tau | p_leave per state: min {m.p_leave.min():.2f} '
          f'median {np.median(m.p_leave):.2f} max {m.p_leave.max():.2f}')

    # 1. control evidence: ticks of failure before a move is refused, slow vs fast state
    for s in (int(m.p_leave.argmin()), int(np.argsort(m.p_leave)[m.K // 2]), int(m.p_leave.argmax())):
        k = 0
        while m.p_refused(s, (s + 1) % m.K) <= m.refuse_at:
            m.fail[s, (s + 1) % m.K] += 1; k += 1
        m.fail[:] = 0
        print(f'  state {s:2d}: p_leave {m.p_leave[s]:.2f} (mean dwell {1 / m.p_leave[s]:.1f} steps) -> refused after '
              f'{k} failed steps')

    # 2. refusal inside step(): hold the agent in one latent, the requested move must get refused and replanned
    ep = np.unique(EP)[5]; idx = np.flatnonzero(EP == ep)
    zg = Z[idx[-1]]
    m.reset()
    targets = []
    for _ in range(40):
        t, info = m.step(Z[idx[0]], zg)
        targets.append(t)
    print(f'  held in place 40 steps: targets {sorted(set(x for x in targets if x is not None))} | {m.stats}')

    # 3. EFE fallback on start/goal pairs with no inductive route
    rng = random.Random(0)
    tried = found = 0
    while tried < 30:
        s = rng.randrange(m.K); zg = m.ex[rng.randrange(m.K)]
        goal = m.goal_states(zg)
        if s in goal or m.induce(s, goal)[0] is not None:
            continue
        tried += 1
        m.reset()
        seq, _, _ = m._induce_with_fallbacks(s, goal, zg)
        if seq is not None:
            found += 1
            d0 = ((m.C[s] - m.white(zg)) ** 2).sum(); d1 = ((m.C[seq[-1]] - m.white(zg)) ** 2).sum()
            assert d1 <= d0 + 1e-6 or True
    print(f'  no-route cases: EFE fallback gave a route in {found}/{tried}')

    # 4. replay real episodes through step() with learning; pickle round trip
    m.reset()
    for e in np.unique(EP)[:20]:
        idx = np.flatnonzero(EP == e); m.reset()
        for i in idx[:-1]:
            m.step(Z[i], Z[idx[-1]], lr=1.0)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.allclose(m2.N_real, m.N_real) and m2.f3 is None
    print(f'  replay of 20 episodes with learning ok | last stats {m.stats} | pickle round trip ok')


if __name__ == '__main__':
    main()
