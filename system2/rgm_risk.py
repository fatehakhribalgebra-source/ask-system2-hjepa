"""Risk planner on a paper-style RGM map (system2/rgm_paper.py): counts -> hop probabilities -> -ln V.

PaperRGM plans by fewest hops over edges that pass the hard 1/16 threshold. Here the same counted map is read as a
probabilistic hop model and the plan is the hop with the highest discounted chance of reaching the goal in the time
left (the risk part of the curious RGM, system2/rgm_curious.py, on the offline / play map):

Hop model. Asking the reflex for s -> t succeeds in a latent step with probability p_st ~ Beta(a, b):
    a = kappa * q_st + made,  b = kappa * (1 - q_st) + failed latent steps (this episode),
    q_st = n_st / (n_st + thresh * m_s),   n_st = counted level-1 transitions s -> t,  m_s = best exit count of s.
  The soft form of the 1/16 rule: an exit at exactly 1/16 of the best exit gets q = 0.5, the best exit 16/17,
  an exit at 1/64 of it 0.2. Self transitions (dwelling) never compete with exits.
Risk. V_k(s) = E[lambda^arrival] of reaching a goal state within k latent steps,
    V_k(s) = lambda * max(V_{k-1}(s), max_t p_st V_{k-1}(t) + (1 - p_st) V_{k-1}(s)),  V(goal) = 1,
  k = latent steps left in the episode (set by the runner through `ticks_left`). Short and probable routes both count;
  the deadline decides between them. The hop handed down maximises ln(p V(t) + (1 - p) V(s)) (+ a small habit bonus
  for the hop already being tried). No route within the time left: plan with a longer horizon; still none: hand over
  to H-JEPA's own planner, as PaperRGM does.
Imagined counts (optional). `add_counts(N, w)` adds weighted extra transitions (e.g. one-step F3 imagination) to n_st.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from system2.rgm_paper import PaperRGM, U_THRESH


class RiskRGM(PaperRGM):
    kappa = 2.0              # prior strength of the hop model, in units of tries
    time_discount = 0.8      # lambda: value of arriving one latent step later
    habit = 0.05             # tie-break toward the hop already being tried
    soft_thresh = U_THRESH   # exit at this fraction of the best exit -> q = 0.5
    ticks_left = 17          # set by the runner each call
    long_horizon = 64        # horizon when nothing is reachable in the time left

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.N_extra = None
        self._build_counts()

    @classmethod
    def from_state(cls, d):
        return super().from_state(d)

    def _build_counts(self):
        lv = self.levels[0]
        N = np.zeros((self.K, self.K))
        for i, c in enumerate(lv.succ):
            s = lv.items[i][0]
            for j, v in c.items():
                t = lv.items[j][0]
                if s != t:
                    N[s, t] += v
        self.N_real = N
        self._refresh()

    def add_counts(self, N, w):
        N = np.asarray(N, np.float64).copy()
        np.fill_diagonal(N, 0.0)
        self.N_extra = w * N if self.N_extra is None else self.N_extra + w * N
        self._refresh()

    def _refresh(self):
        N = self.N_real + (self.N_extra if self.N_extra is not None else 0.0)
        m = N.max(1, keepdims=True)
        self.src, self.dst = np.nonzero(N)
        n = N[self.src, self.dst]
        self.q = n / (n + self.soft_thresh * m[self.src, 0])

    def reset(self):
        super().reset()
        self.ok, self.ko = defaultdict(float), defaultdict(float)
        self.issued = None
        self.stats.update(made=0, failed=0, risk_steps=0, long_route=0, no_route=0)

    def hop_p(self):
        ok = np.array([self.ok.get((s, t), 0.0) for s, t in zip(self.src, self.dst)])
        ko = np.array([self.ko.get((s, t), 0.0) for s, t in zip(self.src, self.dst)])
        a = self.kappa * self.q + ok
        b = self.kappa * (1 - self.q) + ko
        return a / (a + b)

    def reach_prob(self, goal, p, T):
        lam = self.time_discount
        src, dst = self.src, self.dst
        V = np.zeros(self.K)
        V[list(goal)] = 1.0
        Vprev = V.copy()
        for _ in range(max(int(T), 1)):
            Vprev = V
            val = lam * (p * V[dst] + (1 - p) * V[src])
            Vn = lam * V
            np.maximum.at(Vn, src, val)
            Vn[list(goal)] = 1.0
            V = Vn
        return Vprev

    def decide(self, s, goal):
        here = np.flatnonzero(self.src == s)
        if not len(here):
            return None, None
        p = self.hop_p()
        V = self.reach_prob(goal, p, self.ticks_left)
        val = p[here] * V[self.dst[here]] + (1 - p[here]) * V[s]
        tag = 'risk'
        if val.max() <= 0:
            V = self.reach_prob(goal, p, self.long_horizon)
            val = p[here] * V[self.dst[here]] + (1 - p[here]) * V[s]
            tag = 'long_route'
        if val.max() <= 0:
            return None, None
        score = np.log(val + 1e-300)
        if self.last_target is not None:
            score = score + self.habit * (self.dst[here] == self.last_target)
        k = int(score.argmax())
        return int(self.dst[here[k]]), dict(mode=tag, p=round(float(p[here[k]]), 3),
                                            V=round(float(V[self.dst[here[k]]]), 4))

    def step(self, z, zg, observe=True, lr=0.0, tick=True):
        goal = self.goal_states(zg)
        tick = observe and tick
        s_now = self.infer_state(z, tick)
        if tick:
            if self.issued is not None:
                a, t = self.issued
                if s_now == t:
                    self.ok[(a, t)] += 1; self.stats['made'] += 1
                elif s_now == a:                                    # still there: the hop has not happened yet
                    self.ko[(a, t)] += 1; self.stats['failed'] += 1
                else:                                               # went elsewhere: the hop failed
                    self.ko[(a, t)] += 1; self.stats['failed'] += 1
            self.history.append(s_now)
        if s_now in goal:
            self.issued, self.last_target = None, None
            return None, dict(reason='at_goal', s=s_now)
        target, info = self.decide(s_now, goal)
        if target is None:
            self.stats['no_route'] += 1
            self.issued, self.last_target = None, None
            return None, dict(reason='no_path', s=s_now)
        self.stats['risk_steps' if info['mode'] == 'risk' else 'long_route'] += 1
        if tick or self.issued is None:
            self.issued = (s_now, target)
        self.last_target = target
        return target, info | dict(s=s_now, target=target, fit=round(getattr(self, 'last_fit', -1.0), 2))
