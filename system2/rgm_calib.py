"""Calibrated RGM: F3 judges every hop, reality calibrates the judge (two parallel memories).

CuriousRGM prices a hop s -> t from real passive counts, or, for a dreamed hop, from how often similar dreams were
confirmed. In AntMaze only ~30% of the hops it asks for are made: many edges join states that are not one reflex
step apart. Here the world model is asked directly, and reality decides how far to trust it:

Dreamed memory (the world model's verdict). r_st = min over M sampled level-3 macro-actions a of
      || F3(z_s, a) - c_t || / radius,
  i.e. how close one imagined level-3 step from s can get to t, in state radii (0 = F3 says t is one step away).
  Computed once per edge (cached; batched on the GPU).
Seen memory (reality). Real passive transitions e_st counted from the stream, and the outcome of every hop the reflex
  was asked for (made / failed per latent step).
Calibration (world model vs reality). P(hop works | r, e, l) = sigmoid(w0 + w1 r + w2 ln(1 + e) + w3 l), with l the
  logit of the success rate of tried hops NEAR s -> t (kernel in z3; a wall belongs to a place). Bayesian logistic
  regression (MAP, Gaussian prior around w_prior) refit after every tried hop on all tried edges. Its weights say
  how much the agent trusts F3 (w1), its passive counts (w2) and local experience (w3) -- learned, not set. The calibrated value is the
  prior mean q of each edge's Beta; the edge's own tries are added on top, as in CuriousRGM. Risk, novelty and the
  choice are unchanged.
Candidates from geometry. H-JEPA keeps similar situations close in z3, so each real state's geo_k nearest states
  (within geo_r radii) are candidate hops in both directions; F3 and the calibration price them, like any dream.

Without an F3 judge (toy), r falls back to the plain distance in F3-step units.
"""
from __future__ import annotations

import numpy as np

from system2.rgm_curious import CuriousRGM


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class CalibRGM(CuriousRGM):
    geo_k = 0                    # nearest states per real state proposed as candidate hops
    geo_r = 3.0                  # ... if within this many state radii
    w_prior = (0.0, -1.5, 0.5, 1.0)  # bias, F3 verdict r, ln(1+passive), local logit; before any try: r = 0 -> local rate
    local_dir = False            # nearby = same start and same displacement (False: same start and same end)
    local_w = 2.0               # width of "nearby hop" (start and displacement), in state radii
    w_prec = 1.0                # prior precision of the calibration weights (in units of one tried edge-step)
    f3_reach = None              # callable (Zs (n, D), Zt (n, D)) -> min distance of F3 one-step predictions to Zt

    # ------------------------------------------------------------------ the two memories
    def _calib_init(self):
        if not hasattr(self, 'r_f3'):
            self.r_f3, self.geo_done, self.w = {}, set(), np.array(self.w_prior, np.float64)
        if len(self.w) != len(self.w_prior):
            self.w = np.array(self.w_prior, np.float64)

    def _residuals(self, S, T):
        """World model's verdict per edge, in state radii (cached)."""
        self._calib_init()
        need = sorted({(int(s), int(t)) for s, t in zip(S, T)} - self.r_f3.keys())
        if need:
            C = np.stack(self.C)
            Zs = np.stack([self.ex[s] for s, _ in need]); Zt = C[[t for _, t in need]]
            if self.f3_reach is not None:
                d = np.asarray(self.f3_reach(Zs, Zt), np.float64) / self.radius
            else:                                                   # no judge: distance in F3 steps
                d = np.linalg.norm(Zs - Zt, axis=1) / (self.radius / self.rho)
            self.r_f3.update(zip(need, d.tolist()))
        return np.array([self.r_f3[(int(s), int(t))] for s, t in zip(S, T)], np.float64)

    def _tried(self):
        keys = [k for k in set(self.ok) | set(self.ko) if self.ok.get(k, 0) + self.ko.get(k, 0) > 0]
        k = np.array([self.ok.get(x, 0.0) for x in keys], np.float64)
        n = k + np.array([self.ko.get(x, 0.0) for x in keys], np.float64)
        return keys, k, n

    def _local(self, S, T):
        """Logit of the success rate of tried hops NEAR s -> t (start near s, end near t; leave-one-out), shrunk to
        the map-wide rate: a wall or an open corridor is a property of a place, not of one edge."""
        keys, k, n = self._tried()
        gbar = (k.sum() + 1.0) / (n.sum() + 2.0)
        if not keys:
            return np.full(len(S), np.log(gbar / (1 - gbar)))
        C = np.stack(self.C)
        TS = np.array([x[0] for x in keys]); TT = np.array([x[1] for x in keys])
        # similar hop = starts near s AND moves the same way (a wall blocks a direction at a place)
        if self.local_dir:
            D, TD = C[T] - C[S], C[TT] - C[TS]
            d2 = ((C[S][:, None] - C[TS][None]) ** 2).sum(-1) + ((D[:, None] - TD[None]) ** 2).sum(-1)
        else:                                                       # starts near s and ends near t
            d2 = ((C[S][:, None] - C[TS][None]) ** 2).sum(-1) + ((C[T][:, None] - C[TT][None]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * (self.local_w * self.radius) ** 2))
        w[(S[:, None] == TS[None]) & (T[:, None] == TT[None])] = 0.0
        rate = (w @ k + self.kappa * gbar) / (w @ n + self.kappa)
        rate = np.clip(rate, 1e-3, 1 - 1e-3)
        return np.log(rate / (1 - rate))

    def _features(self, S, T):
        S = np.asarray(S, np.int64); T = np.asarray(T, np.int64)
        e = np.array([self.e_real.get((s, t), 0.0) for s, t in zip(S, T)], np.float64)
        return np.stack([np.ones(len(e)), self._residuals(S, T), np.log1p(e), self._local(S, T)], 1)

    def _q(self, S, T):
        S = np.asarray(S, np.int64); T = np.asarray(T, np.int64)
        if not len(S):
            return np.zeros(0)
        self._calib_init()
        return _sig(self._features(S, T) @ self.w)

    def fit_calibration(self):
        """MAP logistic regression of hop outcomes on (1, r, ln(1+e)), over all tried edges (Newton steps)."""
        self._calib_init()
        keys, k, n = self._tried()
        if not keys:
            return
        S = np.array([x[0] for x in keys]); T = np.array([x[1] for x in keys])
        X = self._features(S, T)
        w0 = np.array(self.w_prior, np.float64); w = self.w.copy()
        if len(w) != len(w0):
            w = w0.copy()
        for _ in range(8):
            p = _sig(X @ w)
            g = X.T @ (k - n * p) - self.w_prec * (w - w0)
            H = (X * (n * p * (1 - p))[:, None]).T @ X + self.w_prec * np.eye(len(w))
            step = np.linalg.solve(H, g)
            w = w + step
            if np.abs(step).max() < 1e-4:
                break
        self.w = w

    def _geo_edges(self):
        """Candidate hops from H-JEPA's geometry: near in z3 = maybe one step apart."""
        self._calib_init()
        if self.radius in (None, np.inf) or not self.C:
            return
        new = [s for s in range(self.K) if s not in self.geo_done]      # every new state (real, imagined, goal)
        if not new:
            return
        C = np.stack(self.C)
        for s in new:
            d = np.linalg.norm(C - C[s], axis=1); d[s] = np.inf
            for t in np.argsort(d)[:self.geo_k]:
                if d[t] <= self.geo_r * self.radius:
                    self._edge(s, int(t), dream=1.0); self._edge(int(t), s, dream=1.0)
            self.geo_done.add(s)

    # ------------------------------------------------------------------ hooks
    def infer_state(self, z, tick):
        had = self.issued is not None and tick
        s = super().infer_state(z, tick)
        if had:
            self.fit_calibration()
        return s

    def decide(self, s, goal):
        self._geo_edges()
        return super().decide(s, goal)

    def map_summary(self):
        self._calib_init()
        return super().map_summary() | dict(calib_w=[round(float(x), 3) for x in self.w])

    def __getstate__(self):
        d = super().__getstate__(); d.pop('f3_reach', None)
        return d

    @classmethod
    def adopt(cls, m, f3_reach=None):
        """Turn a CuriousRGM (e.g. an explored map from a pickle) into a CalibRGM, keeping all its memory."""
        m.__class__ = cls
        m.f3_reach = f3_reach
        m._calib_init()
        m.fit_calibration()
        return m
