"""Deep active-inference agent (after Fountas, Sajid, Mediano & Friston, 2020) adapted to play the z3 game.

Game: state z3 (H-JEPA level-3 latent), K discrete moves = real macro-action prototypes, dynamics = H-JEPA's F3.
Agent (its own generative model, in the 16-d state space s of the pretrained state VAE):
  perception    q(s | z3), p(z3 | s)            frozen StateVAE (already trained on real latents)
  transition    p(s' | s, a)                    Gaussian MLP with MC dropout -> parameter uncertainty
  habit         Q(a | s)                        softmax MLP, trained to match MCTS visit distributions (policy prior)
EFE of a move a from s (one tree edge), as in Fountas et al. but with the Free Lunch north star as T(o):
  G(s, a) = - E ln T(o)                         pragmatic (north star; 'typical' = N(0, I) or none)
            - [ H(F(o|s,a)) - E H(p(o|s')) ]    state information gain (mutual information s'/o)
            - [ H(mix over dropout) - E H(each) ]   parameter information gain (novelty about the model)
MCTS: PUCT selection  argmax_a  -Gbar(s,a) + c * Q(a|s) * sqrt(N(s)) / (1 + N(s,a)),  expansion of all K children
in one batch, backup of the summed EFE along the path. Action = from visit counts. The principal variation (most
visited path) is the agent's road.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PIE = math.log(2 * math.pi * math.e)


class DropoutTransition(nn.Module):
    def __init__(self, d_s: int, k: int, hidden: int = 256, p: float = 0.1):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(d_s + k, hidden), nn.SiLU(), nn.Dropout(p), nn.Linear(hidden, hidden),
                                 nn.SiLU(), nn.Dropout(p), nn.Linear(hidden, 2 * d_s))

    def forward(self, s, a_idx):
        mu, lv = self.net(torch.cat([s, F.one_hot(a_idx, self.k).float()], -1)).chunk(2, -1)
        return s + mu, lv.clamp(-8, 2).exp()


class HabitNet(nn.Module):
    def __init__(self, d_s: int, k: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_s, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, k))

    def forward(self, s):
        return self.net(s).log_softmax(-1)


@dataclass
class DeepAIFConfig:
    sims: int = 64
    depth: int = 5
    c_puct: float = 1.0
    dropout_samples: int = 8
    particles: int = 8
    sigma_t: float = 0.5
    w_sig: float = 1.0
    w_param: float = 1.0
    w_prag: float = 1.0
    lr: float = 1e-3
    device: str = 'cuda'


class _Node:
    __slots__ = ('s', 'children', 'N', 'W', 'prior', 'G_edge', 'z_proxy')

    def __init__(self, s):
        self.s, self.children, self.N, self.W, self.prior, self.G_edge = s, None, None, None, None, None


class DeepAIFAgent:
    def __init__(self, svae, protos: torch.Tensor, cfg: DeepAIFConfig = DeepAIFConfig(), seed: int = 0):
        self.svae, self.protos, self.cfg = svae, protos.to(cfg.device), cfg
        self.K = len(protos)
        d_s = svae.d_s
        torch.manual_seed(seed)
        self.trans = DropoutTransition(d_s, self.K).to(cfg.device)
        self.habit = HabitNet(d_s, self.K).to(cfg.device)
        self.opt = torch.optim.Adam(list(self.trans.parameters()) + list(self.habit.parameters()), lr=cfg.lr)
        self.buf_s, self.buf_a, self.buf_s2, self.buf_pi = [], [], [], []
        self.target = None                      # None | 'typical' | callable(depth) -> (d_obs,) target mean

    # ------------------------------------------------------------------ perception
    @torch.no_grad()
    def belief(self, z):
        return self.svae.posterior(z.reshape(1, -1))[0][0]

    # ------------------------------------------------------------------ EFE of all K moves from a state
    @torch.no_grad()
    def edge_efe(self, s, depth: int):
        c, K = self.cfg, self.K
        S = s.reshape(1, -1).expand(K, -1)
        a = torch.arange(K, device=s.device)
        self.trans.train()                                                  # MC dropout on
        mus, vs = [], []
        for _ in range(c.dropout_samples):
            m, v = self.trans(S, a); mus.append(m); vs.append(v)
        self.trans.eval()
        mu, var = torch.stack(mus), torch.stack(vs)                         # (D, K, d_s)
        mbar = mu.mean(0); vtot = var.mean(0) + mu.var(0, unbiased=False)
        p_ig = 0.5 * (LOG2PIE + vtot.log()).sum(-1) - (0.5 * (LOG2PIE + var.log()).sum(-1)).mean(0)
        eps = torch.randn(c.particles, 1, mbar.shape[-1], device=s.device)
        sp = mbar.unsqueeze(0) + vtot.sqrt().unsqueeze(0) * eps
        xm, xv = self.svae.likelihood(sp)
        om, ov = xm.mean(0), xv.mean(0) + xm.var(0, unbiased=False)
        s_ig = 0.5 * (LOG2PIE + ov.log()).sum(-1) - (0.5 * (LOG2PIE + xv.log()).sum(-1)).mean(0)
        if self.target is None:
            prag = torch.zeros(K, device=s.device)
        else:
            if isinstance(self.target, str):
                tm, s2 = torch.zeros_like(om[0]), 1.0
            else:
                tm, s2 = self.target(depth), c.sigma_t ** 2
            prag = -0.5 * (((om - tm) ** 2 + ov) / s2 + math.log(2 * math.pi * s2)).sum(-1)
        G = -c.w_sig * s_ig - c.w_param * p_ig - c.w_prag * prag
        return G, mbar, dict(state_ig=s_ig, param_ig=p_ig, pragmatic=prag)

    # ------------------------------------------------------------------ MCTS
    @torch.no_grad()
    def _expand(self, node, depth):
        G, s_next, _ = self.edge_efe(node.s, depth)
        node.G_edge = G
        node.prior = self.habit(node.s.reshape(1, -1))[0].exp()
        node.children = [_Node(s_next[k]) for k in range(self.K)]
        node.N = torch.zeros(self.K, device=G.device)
        node.W = torch.zeros(self.K, device=G.device)

    @torch.no_grad()
    def search(self, z):
        c = self.cfg
        root = _Node(self.belief(z))
        self._expand(root, 0)
        g_scale = 1.0
        for _ in range(c.sims):
            node, path, depth = root, [], 0
            while node.children is not None and depth < c.depth:
                Qv = torch.where(node.N > 0, -node.W / node.N.clamp_min(1), -node.G_edge)      # value = -EFE
                g_scale = max(g_scale, float(node.G_edge.abs().mean()))
                u = Qv / g_scale + c.c_puct * node.prior * math.sqrt(float(node.N.sum()) + 1) / (1 + node.N)
                k = int(torch.argmax(u))
                path.append((node, k))
                node = node.children[k]; depth += 1
            if node.children is None and depth < c.depth:
                self._expand(node, depth)
            # backup: accumulated EFE along the path from each node down to the leaf (+ best next edge at the leaf)
            leaf_val = float(node.G_edge.min()) if node.G_edge is not None else 0.0
            acc = leaf_val
            for parent, k in reversed(path):
                acc = float(parent.G_edge[k]) + acc
                parent.N[k] += 1; parent.W[k] += acc
        pi = root.N / root.N.sum()
        # principal variation = the agent's road (indices of moves)
        pv, node = [], root
        while node.children is not None and node.N is not None and node.N.sum() > 0:
            k = int(torch.argmax(node.N)); pv.append(k); node = node.children[k]
        _, _, terms = self.edge_efe(root.s, 0)
        return pi, pv, dict(root=root, terms={key: v.detach() for key, v in terms.items()})

    # ------------------------------------------------------------------ learning
    def remember(self, s, a_idx, s2, pi=None):
        self.buf_s.append(s.detach().cpu()); self.buf_a.append(int(a_idx)); self.buf_s2.append(s2.detach().cpu())
        self.buf_pi.append(None if pi is None else pi.detach().cpu())

    def fit(self, steps: int = 20, batch: int = 128):
        if len(self.buf_s) < 8:
            return None
        dev = self.cfg.device
        S = torch.stack(self.buf_s).to(dev); A = torch.tensor(self.buf_a, device=dev); S2 = torch.stack(self.buf_s2).to(dev)
        has_pi = [i for i, p in enumerate(self.buf_pi) if p is not None]
        PI = torch.stack([self.buf_pi[i] for i in has_pi]).to(dev) if has_pi else None
        Sp = S[has_pi] if has_pi else None
        self.trans.train()
        out = {}
        for _ in range(steps):
            idx = torch.randint(len(S), (min(batch, len(S)),), device=dev)
            mu, var = self.trans(S[idx], A[idx])
            nll = (0.5 * ((S2[idx] - mu) ** 2 / var + var.log())).sum(-1).mean()
            loss = nll
            if PI is not None:
                j = torch.randint(len(PI), (min(batch, len(PI)),), device=dev)
                kl = (PI[j] * (PI[j].clamp_min(1e-8).log() - self.habit(Sp[j]))).sum(-1).mean()
                loss = loss + kl
                out['habit_kl'] = float(kl)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            out['trans_nll'] = float(nll)
        self.trans.eval()
        return out

    def state_dict(self):
        return {'trans': self.trans.state_dict(), 'habit': self.habit.state_dict()}

    def load_state_dict(self, d):
        self.trans.load_state_dict(d['trans']); self.habit.load_state_dict(d['habit'])
