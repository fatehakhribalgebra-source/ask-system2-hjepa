"""Connectivity of a PaperRGM map's level-1 graph: strongly connected components, reachability over all state pairs
(directed, any counted edge vs edges passing the 1/16 rule), and the states the eval tasks start and end in.

  python experiments/map_connectivity.py --map data/rgm_stream_curious_B50_seed42.pkl
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_paper import PaperRGM, U_THRESH  # noqa: E402


def reach(A):
    R = A.copy() | np.eye(len(A), dtype=bool)
    for _ in range(len(A)):
        Rn = R | ((R.astype(np.int32) @ A.astype(np.int32)) > 0)
        if (Rn == R).all():
            break
        R = Rn
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', nargs='+', required=True)
    a = ap.parse_args()
    for path in a.map:
        m = PaperRGM.from_state(pickle.load(open(path, 'rb')))
        lv = m.levels[0]
        N = np.zeros((m.K, m.K))
        for i, c in enumerate(lv.succ):
            for j, v in c.items():
                if lv.items[i][0] != lv.items[j][0]:
                    N[lv.items[i][0], lv.items[j][0]] += v
        A_any = N > 0
        mx = N.max(1, keepdims=True)
        A_thr = N > mx * U_THRESH
        sym = (A_any & A_any.T).sum() / max(A_any.sum(), 1)
        out = [f'{pathlib.Path(path).name}: K={m.K}, edges {int(A_any.sum())} ({100 * sym:.0f}% have the reverse edge), '
               f'states with no exit {int((~A_any.any(1)).sum())}, no entry {int((~A_any.any(0)).sum())}']
        for name, A in (('any edge', A_any), ('1/16 edges', A_thr)):
            R = reach(A)
            scc = R & R.T
            sizes = sorted({tuple(np.flatnonzero(r)) for r in scc}, key=len, reverse=True)
            out.append(f'  {name}: reachable ordered pairs {100 * (R.sum() - m.K) / (m.K * (m.K - 1)):.0f}%, '
                       f'largest strongly connected set {len(sizes[0])}/{m.K}')
        print('\n'.join(out))


if __name__ == '__main__':
    main()
