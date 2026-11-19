"""Discrete active inference on a latent place map (pymdp-style, closed-form EFE), for H-JEPA level 3.

Places: k-means clusters of REAL level-3 latents (in the whitened observation space); a place's goal is the mean of
real encodings in it, i.e. a latent H-JEPA can actually reach (unlike multi-step F3 predictions).
Actions: "go to neighbouring place j" (the n_nb nearest places); H-JEPA's planner does the walking.
Generative model: Dirichlet transition counts a[s, j, s'] (prior: you probably end up at j), learned online from
where the ant actually arrives.
EFE (negative value) of choosing j from s, closed form:
  novelty(s, j)   = sum_s' P(s'|s,j) * 0.5 * (1/a[s,j,s'] - 1/sum_s' a[s,j,s'])   (Dirichlet information gain)
  pragmatic(s, j) = cos( c_j - c_s , d_label )   Free Lunch north star: pattern label -> direction on the latent's
                                                  two main axes, order [+PC1, +PC2, -PC1, -PC2] (as N, E, S, W)
  Q(s, j) = w_nov * novelty + w_prag * pragmatic + gamma * sum_s' P(s'|s,j) V(s')     (sophisticated inference,
  V(s) = max_j Q(s, j), iterated `horizon` times)                                      a few steps of value iteration)
"""
from __future__ import annotations

import numpy as np


class PlaceMap:
    def __init__(self, Y: np.ndarray, Z: np.ndarray, n_places: int = 64, n_nb: int = 6, seed: int = 0):
        """Y: (N, d) whitened latents (for clustering / directions); Z: (N, D) raw z3 (for goal centroids)."""
        rng = np.random.default_rng(seed)
        C = Y[rng.choice(len(Y), n_places, replace=False)].copy()
        for _ in range(40):
            lab = ((Y[:, None] - C[None]) ** 2).sum(-1).argmin(1)
            C = np.stack([Y[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(n_places)])
        self.C = C
        self.goal = np.stack([Z[lab == k].mean(0) if (lab == k).any() else Z[rng.integers(len(Z))] for k in range(n_places)])
        self.size = np.bincount(lab, minlength=n_places)
        D = ((C[:, None] - C[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
        self.nb = np.argsort(D, 1)[:, :n_nb]
        self.n = n_places
        self.lab = lab

    def set_experience_neighbours(self, Y_from: np.ndarray, Y_to: np.ndarray, n_nb: int):
        """Neighbours = the places most often actually reached from each place in recorded experience
        (latent nearness is not maze nearness: walls). Places with too few successors keep latent neighbours."""
        a = ((Y_from[:, None] - self.C[None]) ** 2).sum(-1).argmin(1)
        b = ((Y_to[:, None] - self.C[None]) ** 2).sum(-1).argmin(1)
        T = np.zeros((self.n, self.n)); np.add.at(T, (a, b), 1); np.fill_diagonal(T, 0)
        nb = self.nb.copy()
        for s_ in range(self.n):
            order = [j for j in np.argsort(-T[s_]) if T[s_, j] > 0][:n_nb]
            for j in self.nb[s_]:
                if len(order) >= n_nb:
                    break
                if j not in order:
                    order.append(j)
            nb[s_] = order[:n_nb]
        self.nb = nb
        self.T = T

    def place_of(self, y: np.ndarray) -> int:
        return int(((self.C - y.reshape(1, -1)) ** 2).sum(-1).argmin())


class DiscreteAIF:
    def __init__(self, pmap: PlaceMap, w_nov=1.0, w_prag=1.0, gamma=0.9, horizon=3, prior_to_target=1.0,
                 prior_floor=0.02, seed=0):
        self.m, self.w_nov, self.w_prag, self.gamma, self.h = pmap, w_nov, w_prag, gamma, horizon
        n, k = pmap.n, pmap.nb.shape[1]
        self.a = np.full((n, k, n), prior_floor)
        for s in range(n):
            for i, j in enumerate(pmap.nb[s]):
                self.a[s, i, j] += prior_to_target
        self.rng = np.random.default_rng(seed)
        self.visits = np.zeros(n)

    def _terms(self, s, label):
        a = self.a[s]; tot = a.sum(-1, keepdims=True); P = a / tot
        nov = (P * 0.5 * (1 / a - 1 / tot)).sum(-1)                            # (k,)
        if label is None:
            prag = np.zeros(len(nov))
        else:
            d = np.zeros(self.m.C.shape[1]); ax, sg = [(0, 1), (1, 1), (0, -1), (1, -1)][label % 4]; d[ax] = sg
            disp = self.m.C[self.m.nb[s]] - self.m.C[s]
            prag = disp @ d / (np.linalg.norm(disp, axis=1) + 1e-6)
        return P, nov, prag

    def choose(self, s, labels):
        """labels: north-star labels for the next `horizon` decisions (or None). Returns neighbour index i, info."""
        n = self.m.n
        V = np.zeros(n)
        for t in reversed(range(self.h)):                                       # sophisticated inference (value iteration)
            lab = None if labels is None else labels[t]
            Vn = np.zeros(n)
            for x in range(n):
                P, nov, prag = self._terms(x, lab)
                Vn[x] = np.max(self.w_nov * nov + self.w_prag * prag + self.gamma * P @ V)
            V = Vn
        P, nov, prag = self._terms(s, None if labels is None else labels[0])
        Q = self.w_nov * nov + self.w_prag * prag + self.gamma * P @ V
        i = int(np.argmax(Q + 1e-6 * self.rng.standard_normal(len(Q))))
        return i, dict(Q=float(Q[i]), Q_spread=float(Q.std()), novelty=float(nov[i]), pragmatic=float(prag[i]))

    def update(self, s, i, s_arrived):
        self.a[s, i, s_arrived] += 1.0
        self.visits[s_arrived] += 1
