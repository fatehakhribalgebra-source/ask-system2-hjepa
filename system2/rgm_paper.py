"""Paper-style RGM (Friston et al. 2025, "From pixels to planning", Atari section) observing one H-JEPA level.

In the paper the RGM sees game frames, plans by inductive inference at its top level and acts by REFLEX: it predicts
the next observation and the motor system makes it come true; it has no action list. Here the "frame" is the latent of
one H-JEPA level (z3 every 20 env steps, or z2 every 10) and the reflex is the H-JEPA level just below, which receives
the predicted next latent as its subgoal.

Level 1   s     : discrete state of the latent = nearest of K centroids in a 32-d PCA-whitened view. A latent has no
                  pixel layout, so the paper's spatial blocking reduces to a single group.
Level n   item  : a run of 2^(n-1) consecutive level-1 states. Renormalisation in time: a level-n state generates the
                  initial state and path of the level below, which fixes this run. Transitions between consecutive,
                  non-overlapping runs are counted (both phases), as structure learning by counting unique events.
Planning        : inductive inference at the top level (spm_induction): transitions thresholded at 1/16 of the best
                  successor, goal items = items containing a goal state (spm_set_goals), backward propagation for at
                  most 16 steps; the current item is chosen among items starting in the current state on the shortest
                  path. Falls back to lower levels when the top level has no path.
Prediction      : the plan's state sequence; the predicted next observation is the REAL exemplar latent (data medoid)
                  of the next state that differs from the current one -- H-JEPA reaches real encodings reliably.
Commitment      : temporal depth, as in the paper -- the plan is held for one top-level item (2^(n-1) latent steps)
                  while reality matches it (the observed state is one of the next expected states); inductive
                  inference runs again from the real state when the item ends or reality leaves the plan.
Learning        : continual learning of counts at every level from the states the agent actually visits.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

U_THRESH = 1 / 16
MAX_DEPTH = 16


def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C


class Level:
    """Items (tuples of states) with counted transitions item -> next item."""

    def __init__(self, length):
        self.L = length
        self.items: list[tuple] = []
        self.index: dict[tuple, int] = {}
        self.succ: list[Counter] = []
        self.by_start: dict[int, list[int]] = defaultdict(list)

    def item(self, key):
        key = tuple(int(x) for x in key)
        i = self.index.get(key)
        if i is None:
            i = len(self.items)
            self.index[key] = i; self.items.append(key); self.succ.append(Counter())
            self.by_start[key[0]].append(i)
        return i

    def add_sequence(self, states, lr=1.0):
        L = self.L
        for t in range(len(states) - 2 * L + 1):
            a = self.item(states[t:t + L]); b = self.item(states[t + L:t + 2 * L])
            self.succ[a][b] += lr

    def allowed(self, i):
        c = self.succ[i]
        if not c:
            return []
        m = max(c.values())
        return [j for j, v in c.items() if v > m * U_THRESH]

    def backward(self, goal_items):
        """dist[i] = shortest number of item transitions from i to a goal item (<= 16), via thresholded edges."""
        pred = defaultdict(list)
        for i in range(len(self.items)):
            for j in self.allowed(i):
                pred[j].append(i)
        dist = {g: 0 for g in goal_items}
        frontier = list(goal_items)
        for n in range(1, MAX_DEPTH + 1):
            nxt = []
            for j in frontier:
                for i in pred[j]:
                    if i not in dist:
                        dist[i] = n; nxt.append(i)
            if not nxt:
                break
            frontier = nxt
        return dist


class PaperRGM:
    def __init__(self, mean, proj, C, exemplars, tau, levels):
        self.mean, self.proj, self.C, self.ex, self.tau = mean, proj, C, exemplars, tau
        self.levels: list[Level] = levels
        self.K = len(C)
        self.reset()

    # ------------------------------------------------------------------ structure learning
    @classmethod
    def fit(cls, Z, EP, k=64, n_levels=3, d_white=32, seed=0):
        Z = Z.astype(np.float32)
        mean = Z.mean(0)
        ev, V = np.linalg.eigh(np.cov((Z - mean).T))
        proj = (V[:, ::-1][:, :d_white] / np.sqrt(ev[::-1][:d_white])).T.astype(np.float32)   # (d, D)
        Y = (Z - mean) @ proj.T
        C = kmeans(Y, k, seed=seed)
        s = ((Y[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        tau = float(((Y - C[s]) ** 2).sum(-1).mean())
        ex = np.stack([Z[np.flatnonzero(s == j)[((Y[s == j] - C[j]) ** 2).sum(-1).argmin()]] if (s == j).any()
                       else Z[0] for j in range(k)])
        levels = [Level(2 ** n) for n in range(n_levels)]          # item lengths 1, 2, 4, ...
        for e in np.unique(EP):
            seq = s[EP == e]
            for lv in levels:
                lv.add_sequence(seq)
        m = cls(mean, proj, C, ex, tau, levels)
        m.labels = s
        return m

    def white(self, z):
        return (np.asarray(z, np.float32) - self.mean) @ self.proj.T

    def posterior(self, z):
        d2 = ((self.C - self.white(z)[None]) ** 2).sum(-1)
        q = np.exp(-(d2 - d2.min()) / self.tau)
        return q / q.sum()

    def goal_states(self, zg, mass=0.5):
        q = self.posterior(zg)
        order = np.argsort(-q)
        return set(order[: int(np.searchsorted(np.cumsum(q[order]), mass)) + 1].tolist())

    # ------------------------------------------------------------------ inductive planning
    def induce(self, s_now, goal):
        """Plan from state s_now: returns (state sequence starting at s_now, level used, depth) or (None, ...)."""
        for li in range(len(self.levels) - 1, -1, -1):
            lv = self.levels[li]
            goal_items = [i for i, key in enumerate(lv.items) if goal & set(key)]
            if not goal_items:
                continue
            dist = lv.backward(goal_items)
            cand = [i for i in lv.by_start.get(s_now, []) if i in dist]
            if not cand:
                continue
            d0 = min(dist[i] for i in cand)
            cand = [i for i in cand if dist[i] == d0]
            weight = lambda i: sum(lv.succ[i].values()) + 1e-3
            i = max(cand, key=weight)
            seq = list(lv.items[i])
            while dist[i] > 0 and len(seq) < 64:
                nxt = [j for j in lv.allowed(i) if j in dist and dist[j] < dist[i]]
                if not nxt:
                    break
                i = max(nxt, key=lambda j: lv.succ[i][j])
                seq += list(lv.items[i])
            if goal & set(seq):                                     # cut the plan at the first goal state
                seq = seq[: next(t for t, x in enumerate(seq) if x in goal) + 1]
            return seq, li + 1, d0
        return None, 0, -1

    # ------------------------------------------------------------------ acting by prediction (reflex)
    def reset(self):
        self.plan, self.ptr, self.history, self.since = None, 0, [], 0
        self.stats = dict(inductions=0, kept=0)

    def step(self, z, zg, observe=True, lr=0.0):
        """One latent step: observe z, keep or re-induce the plan, return (target state or None at goal, info)."""
        goal = self.goal_states(zg)
        s_now = int(self.posterior(z).argmax())
        if observe:
            self.history.append(s_now)
            if lr > 0:                                              # continual learning from the visited states
                for lv in self.levels:
                    if len(self.history) >= 2 * lv.L:
                        h = self.history[-2 * lv.L:]
                        a = lv.item(h[:lv.L]); b = lv.item(h[lv.L:])
                        lv.succ[a][b] += lr
        if s_now in goal:
            return None, dict(reason='at_goal', s=s_now)
        kept = False
        top_len = self.levels[-1].L
        if self.plan is not None and observe:
            self.since += 1
        if self.plan is not None and self.since < top_len:          # temporal depth: hold one top-level item
            window = self.plan[self.ptr:self.ptr + 3]
            if s_now in window:                                     # reality matches the plan: keep it
                self.ptr += window.index(s_now)
                kept = True
        if not kept:
            seq, lvl, depth = self.induce(s_now, goal)
            if seq is None:
                return None, dict(reason='no_path', s=s_now)
            self.plan, self.ptr, self.since = seq, 0, 0
            self.stats['inductions'] += 1
            info = dict(induced=True, level=lvl, depth=int(depth), plan_len=len(seq))
        else:
            self.stats['kept'] += 1
            info = dict(induced=False)
        rest = self.plan[self.ptr + 1:]
        target = next((x for x in rest if x != s_now), None)
        if target is None:                                          # plan ends here: re-induce next time
            self.plan = None
            target = s_now
        return int(target), info | dict(s=s_now, target=int(target), ptr=self.ptr)

    # ------------------------------------------------------------------ io
    def state_dict(self):
        return dict(mean=self.mean, proj=self.proj, C=self.C, ex=self.ex, tau=self.tau,
                    levels=[dict(L=lv.L, items=lv.items, succ=[dict(c) for c in lv.succ]) for lv in self.levels])

    @classmethod
    def from_state(cls, d):
        levels = []
        for ld in d['levels']:
            lv = Level(ld['L'])
            for key in ld['items']:
                lv.item(key)
            for i, c in enumerate(ld['succ']):
                lv.succ[i] = Counter(c)
            levels.append(lv)
        return cls(d['mean'], d['proj'], d['C'], d['ex'], d['tau'], levels)
