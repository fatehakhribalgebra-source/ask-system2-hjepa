"""RGM with NO offline map: it starts empty and builds everything while it plays (on H-JEPA level 3).

Only the frozen H-JEPA (System 1: encoders, the level-3 predictor F3, levels 2/1 as reflex, level 3 as fallback) is
given. The RGM has no states, no counts and sees no recorded episodes:

States, grown on the fly. The first z3 creates state 0. A z3 farther than `radius` from every state creates a new
  state (new latent cause when the observation fits none); otherwise it updates its state's centre with a running
  mean (e-means style EMA, step 1/n, floor `ema_min`). `radius` = rho x the median length of one F3 step from the
  first observed z3, so the state size is one typical level-3 move, measured without any data. A state's exemplar is
  the REAL latent that created it (H-JEPA's reflex reaches real encodings). The goal image's z3 is real too: it gets
  its own state.
Transitions, learned from experience. Every real level-3 step is counted at the 1/2/4-step scales (items of the
  paper-style RGM, renormalisation in time). F3 imagines one step from the live z3 under macro-actions on the action
  manifold (random directions through level 3's action-encoder LayerNorm: no recorded actions either); an imagined
  next latent that lands inside an existing state adds a down-weighted count. Exits compete with exits, not with
  dwelling (ignore_self).
Control evidence. Every requested hop is an experiment; a move is refused when "impossible" beats "slow" (dwell
  probability learned online per state, Beta(1,1) prior).
Planning. Inductive planning (spm_induction) to the goal state when a route exists; otherwise expected free energy:
  go to the known reachable state nearest the goal (risk) with a novelty bonus; with nothing better than staying,
  hand over to H-JEPA's own planner -- whose trip is still observed and learned from.
Continual. The model is meant to be carried from task to task (eval --rgm-persist).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from system2.rgm_paper import MAX_DEPTH, U_THRESH, Level, PaperRGM


class LiveRGM(PaperRGM):
    live = True
    rho = 1.0                # state radius in units of the median F3 step
    ema_min = 0.02
    f3_weight = 0.1
    novelty_w = 0.5
    p_impossible = 0.1
    refuse_at = 0.9

    def __init__(self, n_levels=3):
        self.levels = [Level(2 ** n) for n in range(n_levels)]
        for lv in self.levels:
            lv.ignore_self = True
        self.C, self.ex, self.n = [], [], []
        self.radius = None
        self.tau = 1.0
        self.stays, self.leaves = defaultdict(float), defaultdict(float)
        self.fail, self.succ_n = defaultdict(float), defaultdict(float)
        self.f3_latents = None                                      # callable z -> (P, D) imagined next latents
        self.reset()

    @property
    def K(self):
        return len(self.C)

    def reset(self):
        self.plan, self.ptr, self.history, self.since = None, 0, [], 0
        self.blocked = set(getattr(self, 'blocked', set()))
        self.issued, self.last_target, self.stuck, self.last_fit = None, None, 0, -1.0
        self.stats = dict(inductions=0, kept=0, new_states=0, imagined=0, refused=0, efe_fallback=0,
                          moves_ok=0, moves_failed=0, no_route=0)

    # ------------------------------------------------------------------ states, grown on the fly
    def nearest(self, z):
        if not self.C:
            return None, np.inf
        d = np.linalg.norm(np.stack(self.C) - z[None], axis=1)
        j = int(d.argmin())
        return j, float(d[j])

    def _calibrate(self, z):
        if self.radius is None:
            if self.f3_latents is not None:
                steps = np.linalg.norm(self.f3_latents(z) - z[None], axis=1)
                self.radius = self.rho * float(np.median(steps))
            else:
                self.radius = np.inf                                # no scale yet: one state until it is learned

    def _new_state(self, z):
        self.C.append(z.copy()); self.ex.append(z.copy()); self.n.append(1.0)
        self.levels[0].item((self.K - 1,))
        self.stats['new_states'] += 1
        return self.K - 1

    def observe(self, z, grow=True):
        z = np.asarray(z, np.float32)
        self._calibrate(z)
        j, d = self.nearest(z)
        if j is None or (grow and d > self.radius):
            return self._new_state(z), 0.0
        if grow:
            self.n[j] += 1
            step = max(1.0 / self.n[j], self.ema_min)
            self.C[j] = self.C[j] + step * (z - self.C[j])
        return j, d

    def goal_states(self, zg, mass=0.5):
        zg = np.asarray(zg, np.float32)
        self._calibrate(zg)
        j, d = self.nearest(zg)
        if j is None or d > self.radius:
            j = self._new_state(zg)
        return {j}

    def white(self, z):
        return np.asarray(z, np.float32)

    # ------------------------------------------------------------------ state inference + learning hooks
    def p_leave(self, a):
        return (self.leaves[a] + 1) / (self.stays[a] + self.leaves[a] + 2)

    def p_refused(self, a, t):
        if self.succ_n[(a, t)] > 0:
            return 0.0
        pi = self.p_impossible
        return pi / (pi + (1 - pi) * (1 - self.p_leave(a)) ** self.fail[(a, t)])

    def infer_state(self, z, tick):
        s, d = self.observe(z, grow=tick)
        self.last_fit = round(d / self.radius, 2) if self.radius not in (None, np.inf) else -1.0
        if tick and self.history:
            a = self.history[-1]
            if a == s:
                self.stays[a] += 1
            else:
                self.leaves[a] += 1
        if tick and self.issued is not None:
            a, t = self.issued
            if s == t:
                self.succ_n[(a, t)] += 1; self.stats['moves_ok'] += 1; self.blocked.discard((a, t))
            elif s == a:
                self.fail[(a, t)] += 1; self.stats['moves_failed'] += 1
                if self.p_refused(a, t) > self.refuse_at and (a, t) not in self.blocked:
                    self.blocked.add((a, t)); self.stats['refused'] += 1; self.plan = None
        if tick and self.f3_latents is not None:                    # dream one step from the live latent
            lv = self.levels[0]
            for zn in self.f3_latents(np.asarray(z, np.float32)):
                j, d = self.nearest(zn)
                if j is not None and j != s and d <= self.radius:
                    lv.succ[lv.item((s,))][lv.item((j,))] += self.f3_weight
                    self.stats['imagined'] += 1
        return s

    # ------------------------------------------------------------------ planning fallback: expected free energy
    def _induce_with_fallbacks(self, s_now, goal, zg):
        seq, lvl, depth = self.induce(s_now, goal, blocked=self.blocked)
        if seq is not None:
            return seq, lvl, depth
        self.stats['no_route'] += 1
        lv = self.levels[0]
        parent = {s_now: None}
        frontier = deque([(s_now, 0)])
        while frontier:
            s, dpt = frontier.popleft()
            i = lv.index.get((s,))
            if i is None or dpt >= MAX_DEPTH:
                continue
            for j in lv.allowed(i, U_THRESH, self.blocked):
                t = lv.items[j][0]
                if t not in parent:
                    parent[t] = s; frontier.append((t, dpt + 1))
        zg = np.asarray(zg, np.float32)
        nmed = float(np.median(self.n)) if self.n else 1.0
        G = {s: float(np.linalg.norm(self.C[s] - zg)) / self.radius - self.novelty_w / (1 + self.n[s] / nmed)
             for s in parent}
        best = min(G, key=G.get)
        if best == s_now or G[best] >= G[s_now] - 0.5:              # no state clearly closer: H-JEPA acts
            return None, 0, -1
        path = [best]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        self.stats['efe_fallback'] += 1
        return path[::-1], 1, len(path) - 1

    def step(self, z, zg, observe=True, lr=1.0, tick=True):
        target, info = super().step(z, zg, observe=observe, lr=lr, tick=tick)
        s = info.get('s')
        self.issued = (s, int(target)) if target is not None and s is not None and target != s else None
        return target, info | dict(K=self.K)

    def exemplar(self, s):
        return self.ex[s]

    def __getstate__(self):
        d = dict(self.__dict__); d['f3_latents'] = None
        return d
