"""Encode every rendered kitchen frame with frozen V-JEPA 2.1 ViT-L (image mode).

Reads data/kitchen/frames256/*.npz, writes data/kitchen/vjepa21L_256/<same name>.npz with
  grid (N, g*g, 1024) float16  -- tokens averaged over a g x g spatial grid (default g = 4)
  mean (N, 1024) float16       -- global token mean
  ep, t, obs, act              -- copied from the frame shard

  python experiments/kitchen/encode_vjepa.py [--grid 4] [--batch 128]
"""
import argparse
import io
import pathlib
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system2.vjepa_encoder import encode, load_encoder, pool_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(ROOT / 'data' / 'kitchen' / 'frames256'))
    ap.add_argument('--dst', default=str(ROOT / 'data' / 'kitchen' / 'vjepa21L_256'))
    ap.add_argument('--grid', type=int, default=4)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--limit', type=int, default=0, help='frames per shard (0 = all); for a quick test')
    a = ap.parse_args()
    dst = pathlib.Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    enc = load_encoder()
    print('unexpected checkpoint keys:', len(getattr(enc, 'unexpected_keys', [])), flush=True)
    for f in sorted(pathlib.Path(a.src).glob('*.npz')):
        out = dst / f.name
        if out.exists() and not a.limit:
            continue
        d = np.load(f, allow_pickle=True)
        n = len(d['jpg']) if not a.limit else min(a.limit, len(d['jpg']))
        G, M = [], []
        t0 = time.time()
        for s in range(0, n, a.batch):
            ims = np.stack([np.array(Image.open(io.BytesIO(b))) for b in d['jpg'][s:s + a.batch]])
            tok = encode(enc, torch.from_numpy(ims))
            G.append(pool_grid(tok, a.grid).half().cpu().numpy())
            M.append(tok.float().mean(1).half().cpu().numpy())
        np.savez(out, grid=np.concatenate(G), mean=np.concatenate(M), ep=d['ep'][:n], t=d['t'][:n],
                 obs=d['obs'][:n], act=d['act'][:n])
        print(f'{f.name}: {n} frames, {n / (time.time() - t0):.0f} frames/s', flush=True)


if __name__ == '__main__':
    main()
