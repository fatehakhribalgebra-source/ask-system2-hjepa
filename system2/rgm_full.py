"""Fuller RGM (Friston et al. 2025 "From pixels to planning"; Zaghw et al. 2026 "RGMs for active inference:
foundations, derivations, and verification") on H-JEPA's level-3 latent, built on the paper-style RGM of
rgm_paper.py, which already has: counted transitions at three temporal scales, inductive planning (spm_induction),
acting by predicting the next real latent (H-JEPA levels 2/1 = reflex) and commitment by temporal depth.

Added here, each switchable so its effect can be measured:

1. State inference by FILTERING (the reference algorithm, Zaghw et al. Sec. 2.2): predict the current state from the
   previous posterior through the posterior-mean transitions mu(B) (Dirichlet counts + prior concentration b0), then
   correct with the likelihood of the current latent:  q_t(s) ~ L(z_t | s) * sum_j mu(B)[s, j] q_{t-1}(j).
   The paper-style RGM decoded each latent on its own; aliased states (one discrete state covering distant places)
   are then corrected by where the agent just was.
2. ACTIVE LEARNING of transitions (Friston et al. eq. 5-7, selection limit of the update gate): a real transition
   observed during the episode is counted only if it increases the mutual information of the level-1 transition
   model; updates that blur it (noise, aliasing jumps) are rejected. Accepted updates enter every level as before.
3. IMAGINED TRANSITIONS from H-JEPA's level-3 predictor F3 (add_transitions): one-step rollouts from real latents
   under real macro-action prototypes, snapped onto the existing states, counted with their own lower weight so
   dreams cannot overrule experience (built by experiments/build_f3_transitions.py).

State and exemplar latents stay REAL (H-JEPA's reflex only reaches real encodings reliably).
"""
from __future__ import annotations

import numpy as np

from system2.rgm_paper import PaperRGM


def mutual_information(N):
    P = N / N.sum()
    pa, pb = P.sum(0, keepdims=True), P.sum(1, keepdims=True)
    m = P > 0
    return float((P[m] * np.log(P[m] / (pb @ pa)[m])).sum())


class FullRGM(PaperRGM):
    filtering = True        # 1. temporal prior from mu(B)
    b0 = 1 / 128            # Dirichlet prior concentration of every transition (as in the paper's Atari demo)
    lik_prec = 1.0          # precision of the latent likelihood (multiplies 1/tau)
    gated = True            # 2. active-learning gate on continual learning

    @classmethod
    def from_paper(cls, d):
        m = cls.from_state(d)
        m.setup()
        return m

    def setup(self):
        """Level-1 transition counts N[next, prev] from the counted single-state items."""
        lv = self.levels[0]
        K = self.K
        N = np.zeros((K, K))
        for i, c in enumerate(lv.succ):
            a = lv.items[i][0]
            for j, v in c.items():
                N[lv.items[j][0], a] += v
        self.N_real = N
        self.N_imag = np.zeros((K, K))
        self.w_imag = 0.0
        self._refresh()
        self.reset()

    def _refresh(self):
        N = self.N_real + self.w_imag * self.N_imag + self.b0
        self.muB = N / N.sum(0, keepdims=True)                      # columns: p(next | prev)

    def add_transitions(self, N_imag, weight):
        """Imagined level-1 transitions (e.g. from F3), counted at `weight` per imagined transition. They extend the
        level-1 planning graph (used when the deeper levels find no path) and the filter's transition prior."""
        self.N_imag, self.w_imag = np.asarray(N_imag, float), float(weight)
        lv = self.levels[0]
        for a, b in zip(*np.nonzero(self.N_imag.T)):                # a = prev, b = next
            ia, ib = lv.item((a,)), lv.item((b,))
            lv.succ[ia][ib] += self.w_imag * self.N_imag[b, a]
        self._refresh()

    def reset(self):
        super().reset()
        self.q = None
        self.stats.update(gate_kept=0, gate_rejected=0, filter_flips=0)

    # ------------------------------------------------------------------ 1. filtering
    def infer_state(self, z, tick):
        d2 = ((self.C - self.white(z)[None]) ** 2).sum(-1)
        self.last_fit = float(d2.min() / self.tau)
        lik = np.exp(-(d2 - d2.min()) * self.lik_prec / self.tau)
        s_alone = int(lik.argmax())
        if not self.filtering or self.q is None:
            q = lik
        else:
            pred = self.muB @ self.q
            if not tick:                                            # a look between latent steps: half a step
                pred = 0.5 * (self.q + pred)
            q = lik * pred
            if q.sum() < 1e-12:                                     # reality contradicts the prior entirely
                q = lik
        q = q / q.sum()
        if tick or self.q is None:
            self.q = q
        s = int(q.argmax())
        self.stats['filter_flips'] += int(s != s_alone)
        return s

    # ------------------------------------------------------------------ 2. active learning
    def learn(self, lr):
        if len(self.history) < 2:
            return
        a, b = self.history[-2], self.history[-1]
        if self.gated:
            N = self.N_real + self.b0
            N2 = N.copy(); N2[b, a] += lr
            if mutual_information(N2) <= mutual_information(N):    # the update blurs the model: reject it
                self.stats['gate_rejected'] += 1
                return
            self.stats['gate_kept'] += 1
        self.N_real[b, a] += lr
        self._refresh()
        super().learn(lr)
