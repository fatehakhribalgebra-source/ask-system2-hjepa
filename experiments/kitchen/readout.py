"""Content check (Content-Geometry paper, Sec. 3): can a linear readout recover the kitchen state from the features?

Ridge regression (alpha chosen on a validation split) from features to each state factor, fit on one set of episodes
and scored (R^2) on held-out episodes. Factors from the 59-d observation:
  arm joints (7), gripper (2), and per object: bottom burner, top burner, light switch, slide cabinet,
  hinge cabinet, microwave, kettle position, kettle orientation.
Feature sets: V-JEPA global mean (1024), V-JEPA 4x4 grid (16k), and a pixel baseline (16x16 grey thumbnails, PCA 256).

  python experiments/kitchen/readout.py --feat data/kitchen/vjepa21L_256 [--frames data/kitchen/frames256]
"""
import argparse
import io
import pathlib

import numpy as np

# FrankaKitchen-v1: env qpos index q >= 9 sits at obs index q + 9 (obs = robot qpos 9 | robot qvel 9 | object qpos 21
# | object qvel 20). OBS_ELEMENT_INDICES (qpos): bottom burner 11-12, top burner 15-16, light switch 17-18,
# slide cabinet 19, hinge cabinet 20-21, microwave 22, kettle 23-29 (pos 3 + quat 4).
OBJ = {'bottom burner': [20, 21], 'top burner': [24, 25], 'light switch': [26, 27], 'slide cabinet': [28],
       'hinge cabinet': [29, 30], 'microwave': [31], 'kettle pos': [32, 33, 34], 'kettle quat': [35, 36, 37, 38]}
FACTORS = {'arm joints': list(range(0, 7)), 'gripper': [7, 8], **OBJ}


def ridge_r2(Xtr, Ytr, Xte, Yte, alphas=(1e-2, 1e-1, 1, 10, 100, 1000)):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = Ytr.mean(0)
    n = len(Xtr); k = int(0.8 * n)
    best, best_a = -np.inf, alphas[0]
    U, S, Vt = np.linalg.svd(Xtr[:k], full_matrices=False)
    for al in alphas:                                           # choose alpha on the last 20% of training rows
        W = Vt.T @ np.diag(S / (S ** 2 + al)) @ U.T @ (Ytr[:k] - ym)
        p = Xtr[k:] @ W + ym
        r = 1 - ((p - Ytr[k:]) ** 2).sum(0) / (((Ytr[k:] - Ytr[k:].mean(0)) ** 2).sum(0) + 1e-9)
        if r.mean() > best:
            best, best_a = r.mean(), al
    U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    W = Vt.T @ np.diag(S / (S ** 2 + best_a)) @ U.T @ (Ytr - ym)
    p = Xte @ W + ym
    r2 = 1 - ((p - Yte) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-9)
    return r2, best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feat', default='data/kitchen/vjepa21L_256')
    ap.add_argument('--frames', default='data/kitchen/frames256')
    ap.add_argument('--shards', type=int, default=2, help='mixed-v2 shards to use (each ~20k frames, 78 episodes)')
    ap.add_argument('--every', type=int, default=4, help='keep every k-th frame')
    a = ap.parse_args()
    F = sorted(pathlib.Path(a.feat).glob('mixed-v2_w*.npz'))[:a.shards]
    mean, grid, obs, ep = [], [], [], []
    for i, f in enumerate(F):
        d = np.load(f)
        s = slice(None, None, a.every)
        mean.append(d['mean'][s].astype(np.float32)); grid.append(d['grid'][s].astype(np.float32).reshape(len(d['mean'][s]), -1))
        obs.append(d['obs'][s]); ep.append(d['ep'][s] + 10_000 * i)
    mean, grid, obs, ep = map(np.concatenate, (mean, grid, obs, ep))
    feats = {'V-JEPA mean': mean, 'V-JEPA 4x4 grid': grid}
    if a.frames:                                               # pixel baseline on the same frames
        from PIL import Image
        px = []
        for i, f in enumerate(F):
            d = np.load(pathlib.Path(a.frames) / f.name, allow_pickle=True)
            px += [np.asarray(Image.open(io.BytesIO(b)).convert('L').resize((16, 16)), np.float32).ravel()
                   for b in d['jpg'][::a.every]]
        feats['pixels 16x16'] = np.stack(px)
    eps = np.unique(ep); rng = np.random.default_rng(0); rng.shuffle(eps)
    te = np.isin(ep, eps[:len(eps) // 4]); tr = ~te
    print(f'{len(ep)} frames, {len(eps)} episodes ({te.sum()} held-out frames)')
    rows = {}
    for name, X in feats.items():
        for fac, cols in FACTORS.items():
            Y = obs[:, cols]
            if Y[tr].std() < 1e-4:
                continue
            r2, al = ridge_r2(X[tr], Y[tr], X[te], Y[te])
            rows.setdefault(fac, {})[name] = float(np.mean(r2))
    names = list(feats)
    print('| factor | ' + ' | '.join(names) + ' |'); print('|---' * (len(names) + 1) + '|')
    for fac, r in rows.items():
        print(f'| {fac} | ' + ' | '.join(f'{r.get(n, float("nan")):.2f}' for n in names) + ' |')


if __name__ == '__main__':
    main()
