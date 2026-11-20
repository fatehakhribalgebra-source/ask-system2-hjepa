"""Full level-2 and level-3 latent sequences of real AntMaze episodes, for the paper-style RGM (system2/rgm_paper.py).

Encoded exactly as experiments/build_macro_action_data.py does (frozen H-JEPA, HJEPA.encode_hierarchical, no random
crops): level 1 = 5 env steps, level 2 = 10 env steps, level 3 = 20 env steps. Saves, per level, the latents in time
order with episode ids and the ant's true xy at each latent step (xy for analysis only; never shown to the agent).

  python experiments/build_latent_seqs.py --episodes 1250
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))
os.environ.setdefault('HJEPA_HOME', str(ROOT / 'data'))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(ROOT / 'data' / 'datasets' / 'visual_antmaze_medium_stitch_val.h5'))
    ap.add_argument('--ckpt', default=str(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed42' / 'ant_hjepa_l3_object.ckpt'))
    ap.add_argument('--episodes', type=int, default=1250)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'latent_seqs_ant_l2_l3.npz'))
    ap.add_argument('--env', default='ant', choices=['ant', 'pusht'])
    ap.add_argument('--start', type=int, default=0, help='first episode (e.g. to sample across a large file)')
    ap.add_argument('--every-ep', type=int, default=1, help='take every k-th episode')
    a = ap.parse_args()

    dev = 'cuda'
    model = torch.load(a.ckpt, map_location='cpu', weights_only=False).to(dev).eval()
    model.interpolate_pos_encoding = True
    for j in model.jepas:
        j.target_length = None
    norm = torch.load(pathlib.Path(a.ckpt).parent / 'normalizer.pt', weights_only=False)['stats']
    am, asd = (np.asarray(norm['action'][k], np.float32).reshape(-1) for k in ('mean', 'std'))
    pm, psd = (np.asarray(norm['proprio'][k], np.float32).reshape(-1) for k in ('mean', 'std'))

    f = h5py.File(a.data, 'r')
    offs, lens = f['ep_offset'][:], f['ep_len'][:]
    FS = 5
    levels = [lv for lv in (2, 3) if lv <= model.num_levels]
    out = {lv: dict(z=[], ep=[], xy=[]) for lv in levels}
    ep_list = list(range(a.start, len(offs), a.every_ep))[: a.episodes]
    for e in ep_list:
        o, n = int(offs[e]), int(lens[e])
        idx = np.arange(0, n - FS + 1, FS)
        if len(idx) < 8:
            continue
        px = torch.from_numpy(f['pixels'][o + idx[0]: o + idx[-1] + 1: FS]).float().div(255)
        px = ((px.permute(0, 3, 1, 2)[None] - IMAGENET_MEAN) / IMAGENET_STD).to(dev)
        if a.env == 'ant':
            qpos, qvel = f['qpos'][o: o + n], f['qvel'][o: o + n]
            prop = (np.concatenate([qpos[idx, 2:], qvel[idx]], 1).astype(np.float32) - pm) / psd
        else:                                                    # Push-T: the dataset's own proprio
            prop = (f['proprio'][o: o + n][idx].astype(np.float32) - pm) / psd
        act = (f['action'][o: o + n].astype(np.float32) - am) / asd
        act = np.stack([act[i: i + FS].reshape(-1) for i in idx])
        info = {'pixels_level1': px, 'proprio_level1': torch.from_numpy(prop)[None].to(dev),
                'action_level1': torch.from_numpy(act)[None].to(dev)}
        with torch.no_grad():
            enc = model.encode_hierarchical(info, sparse_level1_encode=False)
        # analysis only: ant xy, or Push-T block (x, y, angle)
        xy_all = f['xy'][o: o + n] if a.env == 'ant' else f['state'][o: o + n][:, 2:5]
        for lv, every in ((2, 10), (3, 20)):
            if lv not in levels:
                continue
            z = enc[f'embed_{lv}'][0].cpu().numpy()
            t = np.minimum(np.arange(len(z)) * every, n - 1)
            out[lv]['z'].append(z.astype(np.float16)); out[lv]['ep'].append(np.full(len(z), e))
            out[lv]['xy'].append(xy_all[t].astype(np.float32))
        if e % 200 == 0:
            print(f'episode {e}: ' + ' '.join(f'z{lv} {tuple(enc[f"embed_{lv}"].shape)}' for lv in levels), flush=True)
    save = {}
    for lv in levels:
        for k in ('z', 'ep', 'xy'):
            save[f'{k}{lv}'] = np.concatenate(out[lv][k])
    np.savez_compressed(a.out, **save)
    print('saved', a.out, {k: v.shape for k, v in save.items()})

    if a.env != 'ant' or 3 not in levels:
        return
    # consistency: first z3 of episode 0 must match the existing level-3 dataset
    d = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')
    z0, zd = save['z3'][0].astype(np.float32), d['z'][0]
    print(f'check vs macro_actions z3[0]: rel err {np.linalg.norm(z0 - zd) / np.linalg.norm(zd):.4f}')


if __name__ == '__main__':
    main()
