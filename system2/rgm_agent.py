"""RGM agent over H-JEPA level 3, using the RGM's machinery beyond induction (Friston et al. 2025; Zaghw et al. 2026).

Built on FullRGM (filtering through mu(B), gated continual learning, optional F3-imagined transitions) and the
paper-style RGM (three temporal scales, inductive planning, acting by predicting the next real latent, commitment by
temporal depth). Added here:

1. CONTROL EVIDENCE (learning from real z3, in real time). Every hop the RGM asks for is an experiment: at each latent
   step it records whether the requested move (s -> t) happened, or the agent is still in s. Whether a move is
   impossible for the reflex from here is a model comparison: "possible but slow" predicts staying in s with the
   state's normal per-step probability 1 - p_leave(s) (from the data's dwell times); "impossible" predicts staying
   for sure. With a prior p_impossible, the posterior after k failed steps and no success is
        P(impossible | k) = pi / (pi + (1 - pi) (1 - p_leave(s))^k),
   and the move is refused (planned around) above `refuse_at`. A single success proves the move possible. Slow
   states (low p_leave) need many more failures than fast ones, so slow-but-possible moves are kept.
2. SURPRISE AS PRECISION. If the live latent fits its best state worse than `fit_quantile` of the training latents,
   the RGM's state estimate is unreliable: this call is handed to H-JEPA's own planner (and the plan is dropped).
3. CURIOSITY / EXPECTED FREE ENERGY WHEN INDUCTION FAILS. If no route to the goal exists, the agent goes to the
   reachable state (allowed, non-refused level-1 edges, <= 16 steps) with the lowest
        G(s) = risk(s) - novelty_w * novelty(s),  risk = whitened distance to the goal latent / tau,
        novelty = 1 / (1 + n_s / median n),
   instead of handing over at once; it hands over when the best such state is where it already is.
4. F3 IN REAL TIME. When several shortest-route items start from the current state, prefer the one whose first hop
   F3 (one step from the CURRENT z3 under the real macro-action prototypes) predicts is reachable.
5. e-means states (build_rgm_paper.py --cluster emeans) can be used as the level-1 states.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from system2.rgm_full import FullRGM
from system2.rgm_paper import MAX_DEPTH, U_THRESH


class RGMAgent(FullRGM):
    p_impossible = 0.1
    refuse_at = 0.9
    novelty_w = 0.5
    f3_weight = 1.0              # log-odds bonus for a first hop F3 predicts reachable
    use_surprise = True
    use_efe_fallback = True

    def setup(self):
        super().setup()
        for lv in self.levels:                                      # exits compete with exits, not with dwelling
            lv.ignore_self = True
        K = self.K
        self.att = np.zeros((K, K)); self.fail = np.zeros((K, K)); self.succ_n = np.zeros((K, K))
        N = self.N_real
        stay = np.diag(N) / np.maximum(N.sum(0), 1)
        self.p_leave = np.clip(1 - stay, 0.05, 0.9)
        self.n_state = N.sum(0)
        self.fit_thr = np.inf
        self.f3 = None
        self.reset()

    def calibrate(self, Z, quantile=0.995):
        """Surprise threshold: distance of real latents to their nearest state (units of tau)."""
        Y = self.white(np.asarray(Z, np.float32))
        d2 = np.concatenate([((Y[i:i + 4096, None] - self.C[None]) ** 2).sum(-1).min(1) for i in range(0, len(Y), 4096)])
        self.fit_thr = float(np.quantile(d2 / self.tau, quantile))

    def reset(self):
        super().reset()
        self.issued, self.surprised, self.f3_next = None, False, None
        self.stats.update(refused=0, surprise=0, efe_fallback=0, f3_choice=0, moves_ok=0, moves_failed=0)

    # ------------------------------------------------------------------ 1. control evidence
    def p_refused(self, a, t):
        if self.succ_n[a, t] > 0:
            return 0.0
        k = self.fail[a, t]
        pi = self.p_impossible
        return pi / (pi + (1 - pi) * (1 - self.p_leave[a]) ** k)

    def infer_state(self, z, tick):
        s = super().infer_state(z, tick)
        self.surprised = self.use_surprise and self.last_fit > self.fit_thr
        if tick and self.issued is not None:
            a, t = self.issued
            self.att[a, t] += 1
            if s == t:
                self.succ_n[a, t] += 1; self.stats['moves_ok'] += 1
            elif s == a:
                self.fail[a, t] += 1; self.stats['moves_failed'] += 1
                if self.p_refused(a, t) > self.refuse_at and (a, t) not in self.blocked:
                    self.blocked.add((a, t)); self.stats['refused'] += 1
                    self.plan = None                                # replan around the refused move
            if self.succ_n[a, t] > 0:
                self.blocked.discard((a, t))
        return s

    # ------------------------------------------------------------------ 4. F3: rank first hops
    def choose_item(self, lv, cand, s_now):
        if self.f3_next is None or len(cand) < 2:
            return super().choose_item(lv, cand, s_now)

        def score(i):
            hop = next((x for x in lv.items[i] if x != s_now), None)
            if hop is None:                                         # item stays in s_now: look one item ahead
                nxt = lv.allowed(i, U_THRESH, self.blocked)
                hop = next((x for j in nxt for x in lv.items[j] if x != s_now), None)
            bonus = self.f3_weight if hop is not None and hop in self.f3_next else 0.0
            return np.log(sum(lv.succ[i].values()) + 1e-3) + bonus

        best = max(cand, key=score)
        if best != super().choose_item(lv, cand, s_now):
            self.stats['f3_choice'] += 1
        return best

    # ------------------------------------------------------------------ 3. EFE fallback
    def _induce_with_fallbacks(self, s_now, goal, zg):
        seq, lvl, depth = self.induce(s_now, goal, blocked=self.blocked)
        if seq is not None or not self.use_efe_fallback:
            return seq, lvl, depth
        lv = self.levels[0]
        parent = {s_now: None}
        frontier = deque([(s_now, 0)])
        while frontier:
            s, dpt = frontier.popleft()
            if dpt >= MAX_DEPTH:
                continue
            i = lv.index.get((s,))
            if i is None:
                continue
            for j in lv.allowed(i, U_THRESH, self.blocked):
                t = lv.items[j][0]
                if t not in parent:
                    parent[t] = s; frontier.append((t, dpt + 1))
        yg = self.white(zg)
        med = np.median(self.n_state[self.n_state > 0])
        G = {s: float(((self.C[s] - yg) ** 2).sum() / self.tau) - self.novelty_w / (1 + self.n_state[s] / med)
             for s in parent}
        best = min(G, key=G.get)
        if best == s_now:
            return None, 0, -1
        path = [best]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        self.stats['efe_fallback'] += 1
        return path[::-1], 1, len(path) - 1

    # ------------------------------------------------------------------ step
    def step(self, z, zg, observe=True, lr=0.0, tick=True):
        if self.f3 is not None:
            self.f3_next = self.f3(z)
        target, info = super().step(z, zg, observe=observe, lr=lr, tick=tick)
        if self.surprised and target is not None:                  # 2. unreliable state: H-JEPA acts
            self.stats['surprise'] += 1
            self.plan, self.issued = None, None
            return None, dict(reason='surprise', s=info.get('s'), fit=info.get('fit'))
        s = info.get('s')
        self.issued = (s, int(target)) if target is not None and s is not None and target != s else None
        return target, info

    def __getstate__(self):
        d = dict(self.__dict__); d['f3'] = None
        return d
