"""Build (z_top, true level-3 macro-action sequence) pairs from real AntMaze trajectories.

Real ant torques are pushed up through the FROZEN H-JEPA exactly as in training (HJEPA.encode_hierarchical,
random crops disabled): level 1 = 5 env steps (8 torques x 5 stacked, z-scored), level 2 = stride 2, level 3 =
stride 2 -> one level-3 macro-action per 20 env steps. The level-3 latent action (the 4-d input of level 3's
action_embed, i.e. what the planner optimises) is read with a forward hook.
Output: npz with z (N, D) top-level state, acts (N, H, 4) the next H true macro-actions, ep (N,),
z_next (N, H, D) the top-level latents those macro-actions led to (float16).

  python experiments/build_macro_action_data.py --horizon 5 --episodes 1250
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter the dataset is compressed with)
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
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--episodes', type=int, default=1250)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz'))
    a = ap.parse_args()

    dev = 'cuda'
    model = torch.load(a.ckpt, map_location='cpu', weights_only=False).to(dev).eval()
    model.interpolate_pos_encoding = True
    for j in model.jepas:
        j.target_length = None                                   # no random crops: keep every time step
    norm = torch.load(pathlib.Path(a.ckpt).parent / 'normalizer.pt', weights_only=False)['stats']
    am, asd = (np.asarray(norm['action'][k], np.float32).reshape(-1) for k in ('mean', 'std'))
    pm, psd = (np.asarray(norm['proprio'][k], np.float32).reshape(-1) for k in ('mean', 'std'))

    captured = []
    hook = model.jepas[-1].action_embed.register_forward_hook(lambda m, i, o: captured.append(i[0].detach()))

    f = h5py.File(a.data, 'r')
    offs, lens = f['ep_offset'][:], f['ep_len'][:]
    FS, H = 5, a.horizon
    Z, ACT, EP, ZN = [], [], [], []
    for e in range(min(a.episodes, len(offs))):
        o, n = int(offs[e]), int(lens[e])
        idx = np.arange(0, n - FS + 1, FS)                       # level-1 frames, 5 env steps apart
        if len(idx) < 4 * (H + 2):
            continue
        px = torch.from_numpy(f['pixels'][o + idx[0]: o + idx[-1] + 1: FS]).float().div(255)    # (T1,64,64,3)
        px = ((px.permute(0, 3, 1, 2)[None] - IMAGENET_MEAN) / IMAGENET_STD).to(dev)
        qpos, qvel = f['qpos'][o: o + n], f['qvel'][o: o + n]
        prop = (np.concatenate([qpos[idx, 2:], qvel[idx]], 1).astype(np.float32) - pm) / psd
        act = (f['action'][o: o + n].astype(np.float32) - am) / asd
        act = np.stack([act[i: i + FS].reshape(-1) for i in idx])                              # (T1, 40)
        info = {'pixels_level1': px,
                'proprio_level1': torch.from_numpy(prop)[None].to(dev),
                'action_level1': torch.from_numpy(act)[None].to(dev)}
        captured.clear()
        with torch.no_grad():
            out = model.encode_hierarchical(info, sparse_level1_encode=False)
        z3 = out[f'embed_{model.num_levels}'][0]                 # (T3, D)
        a3 = captured[-1][0]                                     # (T3', 4) latent macro-actions
        T = min(len(z3), len(a3))
        for t in range(T - H):
            Z.append(z3[t].cpu().numpy()); ACT.append(a3[t: t + H].cpu().numpy()); EP.append(e)
            ZN.append(z3[t + 1: t + H + 1].cpu().numpy())        # the latents those actions led to
        if e == 0:
            print('episode 0: level-1 steps', len(idx), '| z3', tuple(z3.shape), '| a3', tuple(a3.shape),
                  '| macro-action mean/std', a3.mean().item(), a3.std().item(), flush=True)
        if e % 100 == 0:
            print(f'episode {e}: {len(Z)} pairs', flush=True)
    hook.remove()
    Z, ACT, EP, ZN = np.stack(Z), np.stack(ACT), np.asarray(EP), np.stack(ZN).astype(np.float16)
    np.savez_compressed(a.out, z=Z, acts=ACT, ep=EP, z_next=ZN)
    print('saved', a.out, Z.shape, ACT.shape, 'act mean', ACT.mean(), 'act std per dim', ACT.reshape(-1, 4).std(0))

    # consistency check: z3 of episode 0, frame 0 must equal the planner's own per-observation encoding
    sys.path.insert(0, str(ROOT))
    from system2.hjepa_bridge import build_policy  # noqa: E402
    policy, solver, cfg, _ = build_policy(a.ckpt, 'ant_l3')
    o = int(offs[0])
    info = {'pixels': f['pixels'][o][None, None], 'proprio': f['proprio'][o][None, None].astype(np.float32),
            'qpos': f['qpos'][o][None, None], 'qvel': f['qvel'][o][None, None]}
    with torch.no_grad():
        zp = solver._encode_all_levels(policy._prepare_info(info), key='pixels')[f'embed_{model.num_levels}'].reshape(-1)
    zd = torch.from_numpy(Z[0]).to(zp.device)
    print(f'check: planner z vs dataset z  cos {torch.nn.functional.cosine_similarity(zp, zd, 0).item():.4f}  '
          f'rel err {((zp - zd).norm() / zd.norm()).item():.4f}')


if __name__ == '__main__':
    main()
