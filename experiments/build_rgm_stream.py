"""Build the paper-style RGM (system2/rgm_paper.py) from EXPERIENCE: the per-step z3 streams of explore episodes.

Each explore episode (run_curious.py --stream --dump-stream) saved z3 at every env step. One level-3 step is 20 env
steps, so an episode of T steps gives 20 level-3 sequences by sliding the window one step at a time:
offset o -> z3(o), z3(o+20), z3(o+40), ... Each sequence is counted as its own episode. The states are k-means
clusters in the whitened z3 space (H-JEPA keeps similar situations close and different ones far, so a cluster is a
place); transitions are counted along every sequence. Nothing comes from the offline dataset.

  python experiments/build_rgm_stream.py --streams results/curious/seed42/dump_curious/explore --episodes 50 \
      --out data/rgm_stream_curious_B50.pkl
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


def load_streams(root, n_eps):
    eps = sorted(pathlib.Path(root).glob('ep*'))[:n_eps]
    out = []
    for e in eps:
        fs = sorted(e.glob('stream_task*.npz'))
        if not fs:
            continue
        d = np.load(fs[0])
        step, z = d['step'], d['z'].astype(np.float32)
        last = {int(s): i for i, s in enumerate(step)}               # one latent per env step (last call wins)
        keys = sorted(last)
        out.append((np.array(keys), z[[last[k] for k in keys]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', required=True, help='explore dir with ep*/stream_task*.npz')
    ap.add_argument('--episodes', type=int, default=50, help='use the first N explore episodes')
    ap.add_argument('--stride', type=int, default=20, help='env steps per level-3 step')
    ap.add_argument('--no-slide', action='store_true', help='only offset 0 (one sequence per episode), for comparison')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    streams = load_streams(a.streams, a.episodes)
    Z, EP = [], []
    for ei, (step, z) in enumerate(streams):
        for o in ([0] if a.no_slide else range(a.stride)):
            sel = np.flatnonzero((step - step[0]) % a.stride == o)
            if len(sel) < 2:
                continue
            Z.append(z[sel]); EP.append(np.full(len(sel), ei * a.stride + o))
    Z, EP = np.concatenate(Z), np.concatenate(EP)
    m = PaperRGM.fit(Z, EP, k=a.k, seed=a.seed)
    print(f'{len(streams)} episodes -> {len(np.unique(EP))} sequences, {len(Z)} latents | ' + ' | '.join(
        f'level {i + 1}: {len(lv.items)} items, {sum(len(c) for c in lv.succ)} transitions'
        for i, lv in enumerate(m.levels)), flush=True)
    cnt = np.bincount(m.labels, minlength=a.k)
    print(f'state sizes min {cnt.min()} median {int(np.median(cnt))} max {cnt.max()}', flush=True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'wb') as f:
        pickle.dump(m.state_dict(), f)
    print('saved', a.out)


if __name__ == '__main__':
    main()
