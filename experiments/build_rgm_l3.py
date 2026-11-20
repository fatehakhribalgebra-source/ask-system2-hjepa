"""Build the level-3 RGM (system2/rgm_l3.py) from the real level-3 transitions, with offline checks.

Checks on held-out episodes (structure learned on the other 90%):
  paths  : for start/goal pairs, does inductive planning find a path (level 1 / level 2) and how long is it?
  imagine: closed loop inside H-JEPA's own level-3 predictor F3 -- RGM picks a macro-action prototype, F3 predicts
           the next z3, RGM re-plans -- success if the imagined z3 enters a goal state within --imagine-steps.
           Pairs: (first, last) frame of a held-out episode, i.e. a goal known to be reachable in ~15 level-3 steps.
The final model (saved) is fit on ALL episodes.

  python experiments/build_rgm_l3.py --k 64
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'H-JEPA' / 'h_jepa'))                 # unpickling the H-JEPA checkpoint
from system2.active_inference import StateVAE  # noqa: E402
from system2.rgm_l3 import RGMLevel3  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pairs', type=int, default=100)
    ap.add_argument('--imagine-steps', type=int, default=30)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out or ROOT / 'data' / f'rgm_l3_k{a.k}.pt')
    dev = 'cuda'
    rng = np.random.default_rng(a.seed)

    d = np.load(ROOT / 'data' / 'macro_actions_ant_l3_h5.npz')
    Z, A, EP = d['z'], d['acts'][:, 0].astype(np.float32), d['ep']
    protos = torch.load(ROOT / 'data' / 'z3_game_assets.pt', weights_only=False)['protos'].numpy()
    ck = torch.load(ROOT / 'data' / 'state_vae_ant_l3.pt', map_location=dev, weights_only=False)
    sv = StateVAE(ck['d_z'], ck['d_s'], d_obs=ck.get('d_obs')).to(dev).eval(); sv.load_state_dict(ck['state_dict'])
    with torch.no_grad():
        S = sv.zn(torch.from_numpy(Z).float().to(dev)).cpu().numpy()

    eps = np.unique(EP)
    val_eps = rng.choice(eps, len(eps) // 10, replace=False)
    va = np.isin(EP, val_eps)
    rgm = RGMLevel3.fit(S[~va], A[~va], EP[~va], protos, k=a.k, levels=2, seed=a.seed)
    print(f'structure learning (90% of episodes): K={rgm.K} states, U={rgm.U} paths, '
          f'{int((rgm.B1 > 0).sum())} unique transitions, {len(rgm.events)} level-2 events, '
          f'{int((rgm.B2 > 0).sum())} level-2 transitions', flush=True)

    # ---------------- check 1: are there paths?
    firsts = [np.flatnonzero(EP == e)[0] for e in val_eps]
    lasts = [np.flatnonzero(EP == e)[-1] for e in val_eps]
    pairs = list(zip(firsts, lasts))[: a.pairs]
    for lv in (1, 2):
        m = RGMLevel3.from_state(rgm.state_dict(), levels=lv)
        found, lens, at_goal = 0, [], 0
        for i, j in pairs:
            acts, info = m.plan(S[i], S[j], H=5)
            if info.get('reason') == 'at_goal':
                at_goal += 1
            elif acts is not None:
                found += 1; lens.append(info.get('n', -1))
        print(f'paths, levels={lv}: found {found}/{len(pairs) - at_goal} (start already in goal state: {at_goal}) | '
              f'induction depth mean {np.mean(lens) if lens else float("nan"):.1f}', flush=True)

    # ---------------- check 2: closed loop inside F3
    model = torch.load(ROOT / 'data' / 'ckpts' / 'ant' / 'ant_hjepa_l3' / 'seed42' / 'ant_hjepa_l3_object.ckpt',
                       map_location=dev, weights_only=False).eval()
    top = model.jepas[-1]
    P = torch.from_numpy(protos).float().to(dev)
    for lv in (1, 2):
        m = RGMLevel3.from_state(rgm.state_dict(), levels=lv)
        succ, steps = 0, []
        for i, j in pairs:
            z = torch.from_numpy(Z[i]).float().to(dev).reshape(1, 1, -1)
            yg = S[j]
            ok = False
            for t in range(a.imagine_steps):
                with torch.no_grad():
                    y = sv.zn(z.reshape(1, -1))[0].cpu().numpy()
                acts, info = m.plan(y, yg, H=1)
                if info.get('reason') == 'at_goal':
                    ok = True; steps.append(t); break
                if acts is None:
                    break
                with torch.no_grad():
                    z = top.predict(z, top.action_embed(P[acts[0]].reshape(1, 1, -1)))[:, -1:]
            succ += ok
        print(f'imagined closed loop in F3, levels={lv}: reached goal state {succ}/{len(pairs)} | '
              f'mean imagined level-3 steps {np.mean(steps) if steps else float("nan"):.1f} '
              f'(real episodes: ~{np.mean([np.sum(EP == e) for e in val_eps]):.0f})', flush=True)

    # ---------------- surprise threshold: how unexpected is the next REAL state, normally? (held-out episodes)
    U = ((A[:, None] - protos[None]) ** 2).sum(-1).argmin(1)
    sur = []
    for e in val_eps:
        idx = np.flatnonzero(EP == e)
        for t in range(len(idx) - 1):
            exp = rgm.b1[:, :, U[idx[t]]] @ rgm.posterior(S[idx[t]])
            sur.append(RGMLevel3.surprise(exp / exp.sum(), rgm.posterior(S[idx[t + 1]])))
    sur = np.asarray(sur)
    thresh = float(np.percentile(sur, 90))
    print(f'surprise of real next state (held-out, {len(sur)} transitions): median {np.median(sur):.2f} | '
          f'p75 {np.percentile(sur, 75):.2f} | p90 {thresh:.2f} (threshold) | p99 {np.percentile(sur, 99):.2f} nats')

    # ---------------- expert precision: log-likelihood of the REAL next state under RGM vs F3 (one step, prototype)
    ll_rgm, ll_f3 = [], []
    for e in val_eps:
        idx = np.flatnonzero(EP == e)
        for t in range(len(idx) - 1):
            q_real = rgm.posterior(S[idx[t + 1]])
            exp = rgm.b1[:, :, U[idx[t]]] @ rgm.posterior(S[idx[t]]); exp = exp / exp.sum()
            ll_rgm.append(float(np.log(exp @ q_real + 1e-12)))
            with torch.no_grad():
                z = torch.from_numpy(Z[idx[t]]).float().to(dev).reshape(1, 1, -1)
                zn = top.predict(z, top.action_embed(P[U[idx[t]]].reshape(1, 1, -1)))[:, -1]
                y_im = sv.zn(zn)[0].cpu().numpy()
            ll_f3.append(float(np.log(rgm.posterior(y_im) @ q_real + 1e-12)))
    calib = dict(ll_rgm=float(np.mean(ll_rgm)), ll_f3=float(np.mean(ll_f3)))
    print(f'expert accuracy on real next states (mean log-lik, held-out): RGM {calib["ll_rgm"]:.2f} | '
          f'F3 one-step {calib["ll_f3"]:.2f} | initial w_f3 = exp(diff) {np.exp(calib["ll_f3"] - calib["ll_rgm"]):.2f}')

    final = RGMLevel3.fit(S, A, EP, protos, k=a.k, levels=2, seed=a.seed)
    final.surprise_thresh = thresh
    final.calib = calib
    torch.save(final.state_dict(), out)
    print('saved', out, f'(all episodes: {int((final.B1 > 0).sum())} transitions, {len(final.events)} events)')


if __name__ == '__main__':
    main()
