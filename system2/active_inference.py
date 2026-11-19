"""An active-inference agent whose environment is H-JEPA level 3.

Observation o = z3 (level-3 latent).  Action a = level-3 macro-action (4-d).  Hidden state s (d_s-dim).
Generative model (the agent's, not H-JEPA's):
  likelihood   p(z3 | s)      = N(dec_mu(s), diag dec_var(s))      state VAE decoder (heteroscedastic -> ambiguity)
  posterior    q(s | z3)      = N(enc_mu(z3), diag enc_var(z3))    amortised perception (state VAE encoder)
  transitions  p(s' | s, a)   = ensemble of M diagonal Gaussians   learned ONLINE from the agent's own steps
  preferences  p_C(s)         = N(0, I) (the VAE prior = typical states); a hook for goals later
Policies pi = sequences of H macro-actions, searched with CEM; each scored by expected free energy
  G(pi) = sum_tau [ risk_tau + ambiguity_tau - param_info_gain_tau ]
    risk          KL( q(s_tau | pi) || p_C(s) )
    ambiguity     E_q(s_tau)[ H[ p(z3 | s) ] ]            evaluated at the predicted mean state
    param IG      H[ mixture over heads ] - mean_m H[ head m ]   (information the step would give about the model)
The first action of the best policy is executed in the environment (level 3); the agent observes z3', updates its
belief, learns its transition model, and repeats.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

LOG2PIE = math.log(2 * math.pi * math.e)


class StateVAE(nn.Module):
    def __init__(self, d_z: int, d_s: int = 16, hidden: int = 512, d_obs: int | None = None):
        """d_obs: if set, the agent observes y = proj @ (z - z_mean), a whitened d_obs-dim PCA view of z3 (set
        with set_whitener); otherwise all d_z dims standardised (over-counts correlated dims in ambiguity)."""
        super().__init__()
        self.d_s = d_s
        d_o = d_obs or d_z
        self.register_buffer('z_mean', torch.zeros(d_z)); self.register_buffer('z_std', torch.ones(d_z))
        self.register_buffer('proj', torch.zeros(d_o if d_obs else 0, d_z))
        self.enc = nn.Sequential(nn.Linear(d_o, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * d_s))
        self.dec = nn.Sequential(nn.Linear(d_s, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * d_o))

    def set_whitener(self, Z, k):
        Z = torch.as_tensor(Z, dtype=torch.float64, device=self.z_mean.device)
        mu = Z.mean(0)
        ev, V = torch.linalg.eigh(torch.cov((Z - mu).T))
        self.z_mean.copy_(mu.float()); self.proj.copy_((V.flip(1)[:, :k] / ev.flip(0)[:k].sqrt()).T.float())

    def zn(self, z):
        if len(self.proj):
            return (z - self.z_mean) @ self.proj.T
        return (z - self.z_mean) / self.z_std

    def posterior(self, z):
        mu, lv = self.enc(self.zn(z)).chunk(2, -1)
        return mu, lv.clamp(-8, 4).exp()

    def likelihood(self, s):
        mu, lv = self.dec(s).chunk(2, -1)
        return mu, lv.clamp(-6, 3).exp()                  # in standardised z units

    def loss(self, z, beta=1.0):
        mu, var = self.posterior(z)
        s = mu + var.sqrt() * torch.randn_like(mu)
        xm, xv = self.likelihood(s)
        nll = 0.5 * (((self.zn(z) - xm) ** 2) / xv + xv.log() + math.log(2 * math.pi)).sum(-1).mean()
        kl = 0.5 * (mu ** 2 + var - 1 - var.log()).sum(-1).mean()
        return nll + beta * kl, nll, kl


class TransitionEnsemble(nn.Module):
    def __init__(self, d_s: int, d_a: int = 4, m: int = 5, hidden: int = 256, lr: float = 1e-3):
        super().__init__()
        self.m = m
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d_s + d_a, hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                                                  nn.SiLU(), nn.Linear(hidden, 2 * d_s)) for _ in range(m)])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.S, self.A, self.S2 = [], [], []

    def step(self, s, a):
        """s: (M, N, d_s) per-head states, a: (N, d_a) -> mean, var (M, N, d_s)."""
        mus, vs = [], []
        for i, h in enumerate(self.heads):
            mu, lv = h(torch.cat([s[i], a], -1)).chunk(2, -1)
            mus.append(s[i] + mu); vs.append(lv.clamp(-8, 2).exp())
        return torch.stack(mus), torch.stack(vs)

    def add(self, s, a, s2):
        self.S.append(s.detach().cpu()); self.A.append(a.detach().cpu()); self.S2.append(s2.detach().cpu())

    def fit(self, steps=30, batch=64, device='cuda'):
        if len(self.S) < 2:
            return None
        S, A, S2 = (torch.stack(x).to(device) for x in (self.S, self.A, self.S2))
        lv_ = None
        for _ in range(steps):
            loss = 0.0
            for h in self.heads:
                idx = torch.randint(len(S), (min(batch, len(S)),), device=device)
                mu, lv = h(torch.cat([S[idx], A[idx]], -1)).chunk(2, -1)
                lv = lv.clamp(-8, 2)
                loss = loss + (0.5 * ((S2[idx] - S[idx] - mu) ** 2 / lv.exp() + lv)).mean()
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            lv_ = float(loss) / self.m
        return lv_


@dataclass
class AIFConfig:
    horizon: int = 3
    samples: int = 256
    iters: int = 4
    elites: int = 32
    w_risk: float = 1.0
    w_amb: float = 1.0
    w_ig: float = 1.0
    fit_steps: int = 30
    mode: str = 'eq'               # 'eq' = G = -state IG - pragmatic (- w_param * param IG); 'legacy' = risk+amb-param IG
    w_sig: float = 1.0             # state information gain (epistemic, about hidden states)
    w_prag: float = 1.0            # pragmatic value E ln T(o) (the north star)
    w_param: float = 0.0           # parameter information gain (novelty about the model); off by default
    particles: int = 16
    sigma_t: float = 0.5           # std of the north-star target in whitened observation units
    w_habit: float = 1.0           # policy prior E(pi) = N(a; 0, a_std^2) per step: -ln E added to G (habits)
    device: str = 'cuda'


class ActiveInferenceAgent:
    def __init__(self, svae: StateVAE, trans: TransitionEnsemble, a_std: torch.Tensor, cfg: AIFConfig = AIFConfig(),
                 seed: int = 0, action_map=None):
        """action_map: u -> a onto the valid macro-action manifold (H-JEPA level 3's final action-encoder LayerNorm).
        With it, CEM searches the unconstrained u (prior N(0, I)) and every scored action is a = action_map(u), so F3
        is only ever asked about actions of the kind it was trained on."""
        self.svae, self.trans, self.cfg = svae, trans, cfg
        self.action_map = action_map
        self.a_std = a_std.to(cfg.device).reshape(-1)
        self.gen = torch.Generator(device=cfg.device).manual_seed(seed)
        self.last = None

    @torch.no_grad()
    def belief(self, z):
        return self.svae.posterior(z.reshape(1, -1))                 # (1, d_s) mean, var

    def set_target(self, target):
        """North star T(o): None (no pragmatic term), 'typical' (T = N(0, I) over whitened o, i.e. typical latents),
        or a tensor (H, d_obs) of per-step target means in whitened observation units (std cfg.sigma_t)."""
        self.target = target

    @torch.no_grad()
    def efe(self, s_mu, acts):
        if self.cfg.mode == 'legacy':
            return self.efe_legacy(s_mu, acts)
        return self.efe_eq(s_mu, acts)

    @torch.no_grad()
    def efe_eq(self, s_mu, acts):
        """G(pi) = sum_tau [ - E_F(o|a) KL(F(s|o,a) || F(s|a))  -  E_F(o|a) ln T(o)  -  w_param * param IG ].
        F(s|a): moment-matched ensemble mixture; F(o|a): F(s|a) pushed through the decoder with P particles and
        moment-matched; state information gain = H[F(o|a)] - E_s H[p(o|s)] (Gaussian closed form, = mutual
        information between s and o, = expected KL from prior to posterior over s)."""
        c = self.cfg
        N, H, _ = acts.shape
        M, P = self.trans.m, c.particles
        s = s_mu.expand(N, -1).unsqueeze(0).expand(M, N, -1)
        sig_tot = prag_tot = pig_tot = 0.0
        for t in range(H):
            mu, var = self.trans.step(s, acts[:, t])
            mbar = mu.mean(0)
            vtot = var.mean(0) + mu.var(0, unbiased=False)                               # F(s_tau | pi)
            pig = 0.5 * (LOG2PIE + vtot.log()).sum(-1) - (0.5 * (LOG2PIE + var.log()).sum(-1)).mean(0)
            eps = torch.randn(P, 1, mbar.shape[-1], device=mbar.device, generator=self.gen)
            sp = mbar.unsqueeze(0) + vtot.sqrt().unsqueeze(0) * eps                       # (P, N, d_s) samples of s
            xm, xv = self.svae.likelihood(sp)                                             # p(o | s) per sample
            om = xm.mean(0); ov = xv.mean(0) + xm.var(0, unbiased=False)                  # F(o_tau | pi)
            sig = 0.5 * (LOG2PIE + ov.log()).sum(-1) - (0.5 * (LOG2PIE + xv.log()).sum(-1)).mean(0)
            tgt = getattr(self, 'target', None)
            if tgt is None:
                prag = torch.zeros(N, device=om.device)
            else:
                if isinstance(tgt, str):                                                  # 'typical': T = N(0, I)
                    tm, s2 = torch.zeros_like(om[0]), 1.0
                else:
                    tm, s2 = tgt[min(t, len(tgt) - 1)], c.sigma_t ** 2
                prag = -0.5 * (((om - tm) ** 2 + ov) / s2 + math.log(2 * math.pi * s2)).sum(-1)   # E_F(o) ln T(o)
            sig_tot, prag_tot, pig_tot = sig_tot + sig, prag_tot + prag, pig_tot + pig
            s = mu
        u = getattr(self, '_u', None)
        neg_ln_E = 0.5 * ((u ** 2).sum((-1, -2)) if u is not None else ((acts / self.a_std) ** 2).sum((-1, -2)))
        G = -c.w_sig * sig_tot - c.w_prag * prag_tot - c.w_param * pig_tot + c.w_habit * neg_ln_E
        return G, dict(state_ig=sig_tot, pragmatic=prag_tot, param_ig=pig_tot, ig=sig_tot + c.w_param * pig_tot,
                       neg_ln_E=neg_ln_E)

    @torch.no_grad()
    def efe_legacy(self, s_mu, acts):
        """s_mu: (1, d_s); acts: (N, H, A) -> G (N,), terms dict of (N,)."""
        N, H, _ = acts.shape
        M = self.trans.m
        s = s_mu.expand(N, -1).unsqueeze(0).expand(M, N, -1)
        risk = amb = ig = 0.0
        for t in range(H):
            mu, var = self.trans.step(s, acts[:, t])
            mbar = mu.mean(0)
            vtot = var.mean(0) + mu.var(0, unbiased=False)
            ig_t = 0.5 * (LOG2PIE + vtot.log()).sum(-1) - (0.5 * (LOG2PIE + var.log()).sum(-1)).mean(0)
            risk_t = 0.5 * (vtot + mbar ** 2 - 1 - vtot.log()).sum(-1)
            _, xv = self.svae.likelihood(mbar)
            amb_t = 0.5 * (LOG2PIE + xv.log()).sum(-1)
            risk, amb, ig = risk + risk_t, amb + amb_t, ig + ig_t
            s = mu
        c = self.cfg
        G = c.w_risk * risk + c.w_amb * amb - c.w_ig * ig
        return G, dict(risk=risk, amb=amb, ig=ig)

    @torch.no_grad()
    def road_efe(self, z, acts):
        """EFE of a whole imagined road (acts (N, A)) from z, under the agent's model and current target."""
        s_mu, _ = self.belief(z)
        self._u = None
        G, terms = self.efe(s_mu, acts.unsqueeze(0))
        return float(G[0]), {k: float(v[0]) for k, v in terms.items()}

    @torch.no_grad()
    def act_sample(self, z, temperature: float = 1.0):
        """Like act(), but returns the first action of a policy sampled from q(pi) ∝ exp(-G / temperature) over the
        final CEM population (for diverse imagined roads)."""
        a0, info = self.act(z)
        G, acts = self._last_pop
        k = int(torch.distributions.Categorical(logits=-G / max(temperature, 1e-6)).sample())
        return acts[k, 0], info

    @torch.no_grad()
    def act(self, z):
        """Plan by CEM on EFE; return the first macro-action of the best policy and diagnostics."""
        c = self.cfg
        s_mu, s_var = self.belief(z)
        A = len(self.a_std)
        mu = torch.zeros(c.horizon, A, device=c.device)
        std = (torch.ones(c.horizon, A, device=c.device) if self.action_map is not None
               else self.a_std.expand(c.horizon, A).clone())
        for _ in range(c.iters):
            raw = mu + std * torch.randn(c.samples, c.horizon, A, device=c.device, generator=self.gen)
            acts = self.action_map(raw) if self.action_map is not None else raw
            self._u = raw if self.action_map is not None else None
            G, terms = self.efe(s_mu, acts)
            el = torch.argsort(G)[:c.elites]
            mu, std = raw[el].mean(0), raw[el].std(0) + 1e-3
        k = int(torch.argmin(G))
        self._last_pop = (G, acts)
        self.last = dict(z=z.detach(), s=s_mu[0], a=acts[k, 0].detach(), plan=acts[k].detach())
        info = {key: float(v[k]) for key, v in terms.items()} | dict(
            G=float(G[k]), G_spread=float(G.std()), ig_spread=float(terms['ig'].std()),
            prag_spread=float(terms['pragmatic'].std()) if 'pragmatic' in terms else None,
            a_norm=float(acts[k, 0].norm()))
        return acts[k, 0], info

    def learn(self, z_next, a_true=None, z_from=None):
        """One environment step happened: add (s, a, s') to the transition model and fit."""
        if self.last is None:
            return None
        s = self.last['s'] if z_from is None else self.belief(z_from)[0][0]
        a = self.last['a'] if a_true is None else a_true.reshape(-1)
        s2 = self.belief(z_next)[0][0]
        self.trans.add(s, a, s2)
        return self.trans.fit(steps=self.cfg.fit_steps, device=self.cfg.device)
