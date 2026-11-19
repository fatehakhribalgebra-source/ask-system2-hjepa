"""Train the adapted Fountas model on REAL level-3 transitions (o0, pi0, o1), official training scheme.

o = whitened z3 (the state VAE's 32-d PCA whitener); pi0 = one-hot nearest of the 12 real macro-action prototypes
to the TRUE level-3 macro-action; o1 = the next real z3. Per batch, as in train.py: the transition (mid) is trained
with compute_loss_mid against the encoder's posterior of o1 with precision omega, then the encoder/decoder (down)
with the gamma-mixed KL (0 -> gamma_max after gamma_delay epochs). The habit (top) is trained toward the
softmax(-G) distribution of random goals (as make_batch_dsprites_active_inference does with its reward).

  python experiments/train_fountas_z3.py --epochs 60
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.active_inference import StateVAE  # noqa: E402
from system2.fountas_z3 import ActiveInferenceModelZ3, loss_down  # noqa: E402
import src.torchloss as official_loss  # noqa: E402  (third_party/deep-active-inference-mc, on path via fountas_z3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--s-dim', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=str(ROOT / 'data' / 'fountas_z3.pt'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = 'cuda'
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    svae = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); svae.load_state_dict(ck['state_dict'])
    protos = torch.load(ROOT / 'data' / 'z3_game_assets.pt', map_location=dev, weights_only=False)['protos'].to(dev)
    d = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')
    Z = torch.from_numpy(d['z']).float().to(dev); ZN = torch.from_numpy(d['z_next'][:, 0].astype(np.float32)).to(dev)
    A = torch.from_numpy(d['acts'][:, 0]).float().to(dev); EP = d['ep']
    with torch.no_grad():
        O0, O1 = svae.zn(Z), svae.zn(ZN)
        PI = ((A[:, None] - protos[None]) ** 2).sum(-1).argmin(1)
    K = len(protos)
    val = np.isin(EP, rng.choice(np.unique(EP), len(np.unique(EP)) // 10, replace=False))
    tr = torch.from_numpy(~val).to(dev); va = torch.from_numpy(val).to(dev)
    print(f'train {int(tr.sum())} transitions, val {int(va.sum())} (episode split); action usage', torch.bincount(PI, minlength=K).tolist())

    model = ActiveInferenceModelZ3(s_dim=a.s_dim, o_dim=O0.shape[1], pi_dim=K)
    opt = {'top': torch.optim.Adam(model.model_top.parameters(), lr=1e-4),
           'mid': torch.optim.Adam(model.model_mid.parameters(), lr=1e-4),
           'down': torch.optim.Adam(model.model_down.parameters(), lr=1e-3)}
    var_a, var_d = 1.0, 1.5
    omega = torch.tensor(var_a / 2.0 + var_d, device=dev)       # the official evaluation-time omega
    gamma, gamma_rate, gamma_max, gamma_delay = 0.0, 0.02, 0.8, 10
    idx_tr = torch.nonzero(tr).squeeze(1)
    for ep in range(a.epochs):
        if ep > gamma_delay:
            gamma = min(gamma_max, gamma + gamma_rate)
        model.train(True)
        perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
        for i in range(0, len(perm), a.batch):
            b = perm[i:i + a.batch]
            o0, o1 = O0[b], O1[b]
            pi0 = model.pi_one_hot[PI[b]]
            qs0, _, _ = model.model_down.encoder_with_sample(o0)
            qs1_mean, qs1_logvar = model.model_down.encoder(o1)
            ps1_mean, ps1_logvar = official_loss.train_model_mid(model.model_mid, qs0, qs1_mean, qs1_logvar, pi0,
                                                                 omega, opt['mid'])
            Fd, _ = loss_down(model, o1, ps1_mean.detach(), ps1_logvar.detach(), omega, gamma)
            opt['down'].zero_grad(); Fd.mean().backward(); opt['down'].step()
            # habit: softmax(-G / T) for a random goal from the data (official target construction, temperature 10)
            with torch.no_grad():
                model.goal = O1[idx_tr[int(rng.integers(len(idx_tr)))]]
                s_ = qs0[:16].detach()
                G = torch.stack([model.calculate_G_mean(s_, model.pi_one_hot[k].expand(len(s_), -1))[0] for k in range(K)], 1)
                log_Ppi = torch.log_softmax(-G / 10.0, dim=1)
            official_loss.train_model_top(model.model_top, s_, log_Ppi, opt['top'])
        model.train(False)
        with torch.no_grad():
            s0, _ = model.model_down.encoder(O0[va]); pm, plv = model.model_mid.transition(model.pi_one_hot[PI[va]], s0)
            q1, q1lv = model.model_down.encoder(O1[va]); om, _ = model.model_down.decoder(pm)
            r2_next = 1 - ((om - O1[va]) ** 2).mean() / O1[va].var(0).mean()
            r2_stay = 1 - ((O0[va] - O1[va]) ** 2).mean() / O1[va].var(0).mean()
            rec, _ = model.model_down.decoder(model.model_down.encoder(O1[va])[0])
            r2_rec = 1 - ((rec - O1[va]) ** 2).mean() / O1[va].var(0).mean()
        if ep % 10 == 9 or ep == a.epochs - 1:
            print(f'epoch {ep + 1}: val R^2 reconstruction {r2_rec:.3f} | next-obs prediction {r2_next:.3f} '
                  f'(copy-current baseline {r2_stay:.3f}) | gamma {gamma:.2f}', flush=True)
    torch.save(model.state_dict(), a.out); print('saved', a.out)


if __name__ == '__main__':
    main()
