"""Offline check of the risk planner (system2/rgm_risk.py) on a saved PaperRGM map, numpy only.

For random start / goal states of the map: does the fewest-hop induction (1/16) find a plan, does the risk planner
find one, and how many hops does each take if every handed-down hop is made (the map's own route length)?

  python experiments/test_rgm_risk.py --map checkpoints/rgm/rgm_stream_curious_B50.pkl
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_paper import PaperRGM  # noqa: E402
from system2.rgm_risk import RiskRGM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', required=True)
    ap.add_argument('--pairs', type=int, default=300)
    ap.add_argument('--ticks', type=int, default=17)
    a = ap.parse_args()
    with open(a.map, 'rb') as f:
        st = pickle.load(f)
    pm, rm = PaperRGM.from_state(st), RiskRGM.from_state(st)
    rng = np.random.default_rng(0)
    ind_ok = risk_ok = 0
    ind_len, risk_len = [], []
    for _ in range(a.pairs):
        s, g = rng.choice(pm.K, 2, replace=False)
        seq, _, _ = pm.induce(int(s), {int(g)})
        if seq is not None and g in seq:
            ind_ok += 1
            ind_len.append(len([x for i, x in enumerate(seq) if i == 0 or x != seq[i - 1]]) - 1)
        rm.reset()
        cur, hops = int(s), 0
        for t in range(a.ticks):
            rm.ticks_left = a.ticks - t
            nxt, info = rm.decide(cur, {int(g)})
            if nxt is None:
                break
            cur, hops = nxt, hops + 1
            if cur == g:
                break
        if cur == g:
            risk_ok += 1; risk_len.append(hops)
    print(f'{a.map}: K={pm.K}, edges {len(rm.src)}, q median {np.median(rm.q):.2f} '
          f'(q>0.5: {100 * (rm.q > 0.5).mean():.0f}%)')
    print(f'  induction (1/16): plan reaches the goal {ind_ok}/{a.pairs}, median hops {np.median(ind_len) if ind_len else -1}')
    print(f'  risk planner    : reaches the goal within {a.ticks} hops {risk_ok}/{a.pairs}, '
          f'median hops {np.median(risk_len) if risk_len else -1}')


if __name__ == '__main__':
    main()
