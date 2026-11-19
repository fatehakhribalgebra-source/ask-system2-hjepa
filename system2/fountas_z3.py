"""Fountas et al. (2020) deep active inference, official PyTorch code adapted to H-JEPA level 3 (goal reaching).

Kept from third_party/deep-active-inference-mc (GPL-3.0): ModelTop (habit Q(pi|s)) and ModelMid (transition
p(s1|s0,pi) with MC dropout) used as-is; the EFE decomposition of calculate_G (term0 = expected log preference,
term1 = state entropy terms, term2 = MC-dropout parameter information gain); the training losses of
compute_loss_mid with precision omega; the MCTS algorithm of src/mcts.py (UCT-style selection with W/N,
expansion of all actions, simulation with the habit, back-propagation of G, early stop when one action dominates).
Adapted (the only changes):
  * observations: 64x64 Bernoulli images  ->  whitened 32-d level-3 latent y = PCA-whiten(z3), Gaussian likelihood
    (ModelDown becomes an MLP VAE with a heteroscedastic Gaussian decoder; entropies/log-liks are Gaussian)
  * actions: 4 dSprites moves -> K = 12 real level-3 macro-action prototypes (pi one-hot over K); no inverse-move
    trimming (no known inverses)
  * preference: the fixed dSprites reward -> goal preference T(o) = N(o; y_goal, sigma^2), y_goal = the encoded goal
"""
from __future__ import annotations

import math
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = pathlib.Path(__file__).resolve().parents[1] / 'third_party' / 'deep-active-inference-mc'
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.torchmodel import ModelMid, ModelTop  # noqa: E402  (official, unchanged)
from src.torchutils import entropy_normal_from_logvar, kl_div_loss_analytically_from_logvar_and_precision  # noqa: E402

LOG2PI = math.log(2 * math.pi)


class ModelDownZ3(nn.Module):
    """MLP replacement for the conv VAE: q(s|o) and p(o|s) with o = whitened z3 (Gaussian)."""

    def __init__(self, s_dim: int, o_dim: int, hidden: int = 256, p: float = 0.1):   # official 0.5 is for images
        super().__init__()
        self.s_dim, self.o_dim = s_dim, o_dim
        self.qs_net = nn.Sequential(nn.Linear(o_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, hidden),
                                    nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, s_dim * 2))
        self.po_net = nn.Sequential(nn.Linear(s_dim, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, hidden),
                                    nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, o_dim * 2))

    @staticmethod
    def reparameterize(mean, logvar):
        return torch.randn_like(mean) * torch.exp(0.5 * logvar) + mean

    def encoder(self, o):
        m, lv = torch.split(self.qs_net(o), self.s_dim, dim=1)
        return m, lv.clamp(-8, 4)

    def encoder_with_sample(self, o):
        m, lv = self.encoder(o)
        return self.reparameterize(m, lv), m, lv

    def decoder(self, s):
        m, lv = torch.split(self.po_net(s), self.o_dim, dim=1)
        return m, lv.clamp(-6, 3)


class ActiveInferenceModelZ3:
    def __init__(self, s_dim=16, o_dim=32, pi_dim=12, sigma_goal=1.0, device='cuda'):
        self.device = torch.device(device)
        self.s_dim, self.o_dim, self.pi_dim = s_dim, o_dim, pi_dim
        self.model_top = ModelTop(s_dim, pi_dim).to(self.device)
        self.model_mid = ModelMid(s_dim, pi_dim).to(self.device)
        self.model_down = ModelDownZ3(s_dim, o_dim).to(self.device)
        self.pi_one_hot = torch.eye(pi_dim, device=self.device)
        self.precision = torch.float32
        self.sigma_goal = sigma_goal
        self.goal = None                                     # (o_dim,) whitened goal latent

    def modules(self):
        return [self.model_top, self.model_mid, self.model_down]

    def train(self, mode=True):
        for m in self.modules():
            m.train(mode)

    def state_dict(self):
        return {'top': self.model_top.state_dict(), 'mid': self.model_mid.state_dict(),
                'down': self.model_down.state_dict(), 's_dim': self.s_dim, 'o_dim': self.o_dim, 'pi_dim': self.pi_dim}

    def load_state_dict(self, d):
        self.model_top.load_state_dict(d['top']); self.model_mid.load_state_dict(d['mid'])
        self.model_down.load_state_dict(d['down'])

    # ---- preference (replaces check_reward): expected log T(o) under the predicted Gaussian observation
    def check_reward(self, po_mean, po_logvar):
        s2 = self.sigma_goal ** 2
        g = self.goal.reshape(1, -1)
        return -0.5 * (((po_mean - g) ** 2 + po_logvar.exp()) / s2 + math.log(2 * math.pi * s2)).sum(-1)

    @staticmethod
    def entropy_obs(po_logvar):                               # replaces entropy_bernoulli (Gaussian)
        return entropy_normal_from_logvar(po_logvar).sum(-1)

    # ---- EFE, same three terms as the official calculate_G_mean / calculate_G_given_trajectory
    @torch.no_grad()
    def calculate_G_mean(self, s0, pi0):
        self.model_mid.train()                                # MC dropout on (as in the official sampler)
        _, ps1_mean, ps1_logvar = self.model_mid.transition_with_sample(pi0, s0)
        po1_m, po1_lv = self.model_down.decoder(ps1_mean)
        _, qs1_mean, qs1_logvar = self.model_down.encoder_with_sample(po1_m)
        term0 = self.check_reward(po1_m, po1_lv)                                         # E log P(o|C)
        term1 = -torch.sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), dim=1)
        n = getattr(self, 'ig_samples', 1)                     # official: 1 sample (noisy); average over n
        term2 = 0.0
        for _ in range(n):
            _, po_lv_theta = self.model_down.decoder(self.model_mid.transition_with_sample(pi0, s0)[1])
            _, po_lv_s = self.model_down.decoder(self.model_down.reparameterize(ps1_mean, ps1_logvar))
            term2 = term2 + (self.entropy_obs(po_lv_theta) - self.entropy_obs(po_lv_s)) / n   # dropout info gain
        self.model_mid.eval()
        return -term0 + term1 + term2, [term0, term1, term2], ps1_mean

    @torch.no_grad()
    def mcts_step_simulate(self, starting_s, depth):
        """Official simulation: roll the habit policy for `depth` steps, average G along the trajectory."""
        s, G = starting_s.unsqueeze(0), 0.0
        for _ in range(depth):
            if getattr(self, 'sim_mode', 'habit') == 'greedy':     # official 'single-step EFE' policy for rollouts
                gk, _, sk = self.calculate_G_mean(s.expand(self.pi_dim, -1), self.pi_one_hot)
                k = int(torch.argmin(gk)); g, s = gk[k:k + 1], sk[k:k + 1]
            else:                                                   # official: sample from the habit Q(pi|s)
                q = self.model_top.encode_s(s)[1][0]
                pi = torch.zeros(1, self.pi_dim, device=self.device); pi[0, torch.multinomial(q, 1)] = 1.0
                g, _, s = self.calculate_G_mean(s, pi)
            G += float(g.mean())
        return G / depth, self.model_top.encode_s(starting_s.unsqueeze(0))[1][0]


class Node:
    """src/mcts.py Node, generalised from pi_dim in {3, 4} to any K (no inverse-move trimming)."""

    def __init__(self, s, model, C, using_prior_for_exploration=False):
        self.K = model.pi_dim
        self.s = torch.stack([s] * self.K)
        self.model, self.C, self.prior_explore = model, C, using_prior_for_exploration
        self.W = torch.zeros(self.K); self.N = torch.zeros(self.K); self.Qpi = torch.full((self.K,), 1.0 / self.K)
        self.children = [None] * self.K
        self.in_progress = -1

    def Q(self):
        return self.W / self.N

    def probs_for_selection(self):
        q = self.Q(); q = q - q.min(); q = q / (q.sum() + 1e-12)
        return q + self.C * (self.Qpi if self.prior_explore else 1.0) / self.N

    def select(self):
        path, acts = [], []
        self.in_progress = int(torch.argmax(self.probs_for_selection()))
        acts.append(self.in_progress); path.append(self.children[self.in_progress])
        while None not in path[-1].children:
            node = path[-1]; node.in_progress = int(torch.argmax(node.probs_for_selection()))
            acts.append(node.in_progress); path.append(node.children[node.in_progress])
        return path, acts

    def expand(self):
        G, _, s_next = self.model.calculate_G_mean(self.s, self.model.pi_one_hot)
        self.W -= G.detach().cpu(); self.N += 1.0
        for i in range(self.K):
            self.children[i] = Node(s_next[i], self.model, self.C, self.prior_explore)

    def backpropagate(self, path, G):
        for node in path:
            node.W[node.in_progress] -= G; node.N[node.in_progress] += 1; node.in_progress = -2

    def action_selection(self):
        path = [int(torch.argmax(self.N))]
        node = self.children[path[-1]]
        while None not in node.children:
            path.append(int(torch.argmax(node.N))); node = node.children[path[-1]]
        return path


class MCTSParams:                      # official defaults (src/mcts.py MCTS_Params)
    C = 1.0
    threshold = 0.5
    repeats = 300
    simulation_depth = 3
    using_prior_for_exploration = False


@torch.no_grad()
def active_inference_mcts(model: ActiveInferenceModelZ3, o, params=MCTSParams()):
    """Official active_inference_mcts, on a whitened z3 observation o (o_dim,). Returns the action path."""
    model.train(False)
    qs0_mean, _ = model.model_down.encoder(o.reshape(1, -1))
    root = Node(qs0_mean[0], model, params.C, params.using_prior_for_exploration)
    root.Qpi = model.model_top.encode_s(qs0_mean)[1][0].cpu()
    root.expand()
    for repeat in range(params.repeats):
        Nn = root.N / root.N.sum()
        if float(Nn.max() - Nn.mean()) > params.threshold:          # calc_threshold: one action dominates
            return root.action_selection(), repeat
        path, _ = root.select()
        path[-1].expand()
        G, path[-1].Qpi = model.mcts_step_simulate(path[-1].s[0], params.simulation_depth)
        path[-1].Qpi = path[-1].Qpi.cpu()
        root.backpropagate([root] + path[:-1], G)
    return root.action_selection(), params.repeats


# ---------------------------------------------------------------- training (official losses, Gaussian obs)
def loss_down(model, o1, ps1_mean, ps1_logvar, omega, gamma, beta_o=1.0, beta_s=1.0):
    md = model.model_down
    qs1_mean, qs1_logvar = md.encoder(o1)
    qs1 = md.reparameterize(qs1_mean, qs1_logvar)
    po_m, po_lv = md.decoder(qs1)
    logpo1_s1 = (-0.5 * ((o1 - po_m) ** 2 / po_lv.exp() + po_lv + LOG2PI)).sum(-1)       # Gaussian, not Bernoulli
    zero = torch.zeros_like(qs1_mean)                     # official port passes floats here (crashes in torch.exp)
    kl_naive = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, zero, zero, omega).sum(1)
    kl_s = kl_div_loss_analytically_from_logvar_and_precision(qs1_mean, qs1_logvar, ps1_mean, ps1_logvar, omega).sum(1)
    return -beta_o * logpo1_s1 + beta_s * (gamma * kl_s + (1.0 - gamma) * kl_naive), (-logpo1_s1).mean()
