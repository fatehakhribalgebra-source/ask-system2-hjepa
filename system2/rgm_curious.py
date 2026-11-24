"""Curious RGM: the map is built by playing + imagining, driven by curiosity; one expected-free-energy planner.

Built on LiveRGM (states grown on the fly from real z3, real exemplars, F3 one-step imagination, dreaming). What is
new here replaces the paper's fixed machinery (the 1/16 threshold, fewest-hop induction, depth 16, the fallback
cascade and the refusal rule) with one probabilistic model and one objective:

Hop model (control-conditioned, Beta per edge). Asking the reflex for the hop s -> t is an experiment that succeeds
  in a given latent step with probability p_st ~ Beta(a, b):
      a = kappa * q_st + successes,  b = kappa * (1 - q_st) + failed latent steps,
      q_st = e / (e + 1) with e = real passive transitions s -> t, if any;
      q_st = r_st (learned precision of similar dreams) if the edge was only imagined.
  Dream evidence does not accumulate: imagining the same move again adds no information, so it never outweighs
  reality. r_st = rate at which reality confirmed tried dreamed hops that start near s and end near t (Gaussian kernel),
  shrunk toward the map-wide rate: the precision of imagination is learned from reality, and it generalises across
  space (after a few dreams through a wall are refuted, every dream crossing there loses trust; in open space dreams
  keep their weight), and it is learned per rollout depth (1 = one F3 step from a real latent, 2, 3+), since long
  rollouts drift. A dreamed edge through a wall is tried, fails, and its p falls: no separate refusal rule.
Risk (pragmatic). V_k(s) = E[lambda^(arrival time)] of reaching a goal state within k latent steps, best hop sequence,
      V_k(s) = lambda * max(V_{k-1}(s), max_t p_st V_{k-1}(t) + (1 - p_st) V_{k-1}(s)),   V(goal) = 1,
  with k = the latent steps left in the episode and lambda = time_discount. Short routes AND probable routes both
  raise V (a failed try costs a step): a long certain route can beat a short doubtful one when time allows, and not
  when it does not. Risk = -ln V; one latent step of delay costs -ln lambda nats.
Novelty (epistemic). Expected information gain of one try of s -> t about p_st (mutual information between the
  outcome and p under the Beta posterior, closed form with digamma). It is high for untested and dreamed edges and
  fades as they are tried, so it cannot fixate on noise. U_k(s) = best discounted sum of information gain along a
  route (same hop model), so the agent will travel to the frontier, not only look at its own exits.
Choice. Each latent step the hop from the current state maximises
      ln q(t) + eta * IG(s, t)   if a goal is reachable (q(t) = discounted reach probability via t; EFE of this hop),
      E(t)                       otherwise (no goal given = explore; goal not on the map yet = explore until it is),
  with E(t) = IG(s, t) + gamma * (p U(t) + (1 - p) U(s)). With eta = 1 risk and novelty are both in nats; with
  lambda = 0.8 an untested hop (IG ~ 0.2) is worth about one step of detour. With nothing left to learn and no route, explore hands down a
  random F3 step (a new imagined state) and goal mode hands over to H-JEPA, whose trip is still learned.
Imagination aimed at the frontier. One-step F3 imagination every latent step may create up to `new_imag_per_tick`
  imagined states beyond the map; sleep (task start) dreams F3 rollouts from the states with the most to learn.

Remaining constants are interpretable: kappa (prior strength), kernel_w (how far a test generalises), eta
(epistemic precision), lambda (time preference), gamma (discount of future information), rho (state size in F3
steps).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from system2.rgm_live import LiveRGM


def digamma(x):
    """Digamma for positive arrays (recurrence to x >= 6, then the asymptotic series)."""
    x = np.asarray(x, np.float64).copy()
    out = np.zeros_like(x)
    while True:
        small = x < 6
        if not small.any():
            break
        out[small] -= 1.0 / x[small]
        x[small] += 1.0
    f = 1.0 / (x * x)
    return out + np.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))))


def beta_bernoulli_ig(a, b):
    """Mutual information (nats) between one Bernoulli outcome and its rate p ~ Beta(a, b)."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    n = a + b
    m = a / n
    h_mean = -(m * np.log(m + 1e-300) + (1 - m) * np.log(1 - m + 1e-300))
    e_h = -(m * (digamma(a + 1) - digamma(n + 1)) + (1 - m) * (digamma(b + 1) - digamma(n + 1)))
    return np.maximum(h_mean - e_h, 0.0)


class CuriousRGM(LiveRGM):
    rho = 0.5                # state radius in units of the median F3 step (LiveRGM used 1.0: too coarse, failed hops)
    kappa = 2.0              # prior strength of the hop model, in units of real tries
    kernel_w = 2.0           # how far a dream test generalises, in state radii (start and end of the hop)
    eta = 1.0                # epistemic precision when a goal is reachable
    time_discount = 0.8      # lambda: value of arriving one latent step later
    gamma = 0.9              # discount of future information gain
    u_horizon = 8            # latent steps of look-ahead for information gain
    ig_eps = 1e-3            # below this nothing is worth learning here
    habit = 0.05             # tie-break toward the hop already being tried
    new_imag_per_tick = 2    # imagined states one-step imagination may add per latent step
    frontier_seeds = 8       # real states dreamed from at sleep (the ones with the most to learn)
    explore = False          # True: no goal (phase 1), curiosity only
    explorer = 'curious'     # explore phase: 'curious' (EFE novelty) or 'random' (random F3 step each latent step)
    ticks_left = 17          # set by the runner each call (latent steps left in the episode)
    fresh_f3_targets = True  # imagined target -> fresh one-step F3 latent from now (False: its stored exemplar)
    hit_radius = True        # a hop counts as made within one radius of the target's centre (False: nearest only)
    stride = 20              # env steps per level-3 step (sliding-window transitions in record())

    def __init__(self, n_levels=1):
        super().__init__(n_levels=n_levels)
        self.e_real, self.e_dream = defaultdict(float), defaultdict(float)
        self.ok, self.ko = defaultdict(float), defaultdict(float)
        self.nbr = defaultdict(set)
        self.d_tests = []                                       # (s, t, confirmed, depth): first tries of dream-only hops
        self.d_depth = {}                                       # shallowest rollout depth an edge was dreamed at
        self.d_ok_all = self.d_try_all = 0.0
        self.rng = np.random.default_rng(0)

    def reset(self):
        super().reset()
        self.stats.update(pragmatic=0, epistemic=0, random_step=0, handover=0, at_goal=0, stream_steps=0,
                          stream_edges=0)
        self._new_this_tick = 0
        self._stream, self._visited = {}, set()

    # ------------------------------------------------------------------ every env step (sliding window)
    def record(self, step, z):
        """One env step's z3. Each H-JEPA level encodes a single frame and one level-3 step spans `stride` env steps,
        so every env step t gives a real transition state(z3[t - stride]) -> state(z3[t]): a sliding window with 20x
        the evidence of one transition per latent step. Every frame also grows / refines states, and a state's
        exemplar becomes the real latent closest to its centre (H-JEPA keeps similar situations close in z3, so the
        centre is the typical one)."""
        if not hasattr(self, '_stream'):
            self._stream, self._visited = {}, set()
        z = np.asarray(z, np.float32)
        self._calibrate(z)
        s, _ = self.observe(z, grow=True)
        if self.real[s] and np.linalg.norm(z - self.C[s]) < np.linalg.norm(self.ex[s] - self.C[s]):
            self.ex[s] = z.copy()
        self._stream[int(step)] = s
        self._visited.add(s)
        self.stats['stream_steps'] += 1
        s0 = self._stream.pop(int(step) - self.stride, None)
        if s0 is not None and s0 != s:
            self._edge(s0, s, real=1.0)
            self.stats['stream_edges'] += 1

    # ------------------------------------------------------------------ evidence
    def _edge(self, s, t, real=0.0, dream=0.0, depth=1):
        if s == t:
            return
        self.nbr[s].add(t)
        if real:
            self.e_real[(s, t)] += real
        if dream:
            self.e_dream[(s, t)] += dream
            b = min(int(depth), 3)                                  # 1 = one F3 step from a real latent, 3 = 3+
            self.d_depth[(s, t)] = min(self.d_depth.get((s, t), 3), b)

    def dream_prior(self, S, T):
        """Learned precision of imagination for the dreamed hops S -> T (arrays): the rate at which reality confirmed
        SIMILAR dreamed hops (start near S and end near T, Gaussian kernel of width kernel_w state radii), shrunk
        toward the map-wide rate (Beta(1, 1) to start). A wall is a property of space, not of one edge: a refuted
        dream lowers the trust in every dream that crosses the same place."""
        S = np.asarray(S, np.int64); T = np.asarray(T, np.int64)
        D = np.array([self.d_depth.get((s, t), 3) for s, t in zip(S, T)], np.int64)
        tests = np.asarray(self.d_tests, np.float64).reshape(-1, 4)
        # map-wide rate per rollout depth (long rollouts drift: their trust is learned separately). A d-step dream is
        # right only if every step is: before its own tests, depth d starts at g1^d (g1 = the one-step rate).
        n1, k1 = (tests[:, 3] == 1).sum(), tests[tests[:, 3] == 1, 2].sum()
        g1 = (k1 + 1.0) / (n1 + 2.0)
        g = np.array([(tests[tests[:, 3] == d, 2].sum() + self.kappa * g1 ** d) / ((tests[:, 3] == d).sum() + self.kappa)
                      if d > 1 else g1 for d in (1, 2, 3)])
        prior = g[D - 1]
        if not len(tests) or not len(S):
            return prior
        C = np.stack(self.C)
        TS, TT, TO, TD = (tests[:, 0].astype(np.int64), tests[:, 1].astype(np.int64), tests[:, 2],
                          tests[:, 3].astype(np.int64))
        sig2 = 2 * (self.kernel_w * self.radius) ** 2
        d2 = (((C[S][:, None] - C[TS][None]) ** 2).sum(-1) + ((C[T][:, None] - C[TT][None]) ** 2).sum(-1))
        w = np.exp(-d2 / sig2) * (D[:, None] == TD[None])
        w[(S[:, None] == TS[None]) & (T[:, None] == TT[None])] = 0.0   # an edge's own test is in its own Beta
        return (w @ TO + self.kappa * prior) / (w.sum(1) + self.kappa)

    def _q(self, S, T):
        """Prior probability that the hop works, before its own tries: real passive evidence, else the dream prior."""
        S = np.asarray(S, np.int64); T = np.asarray(T, np.int64)
        e = np.array([self.e_real.get((s, t), 0.0) for s, t in zip(S, T)], np.float64)
        dreamed = np.array([self.e_dream.get((s, t), 0.0) > 0 for s, t in zip(S, T)], bool)
        q = np.where(e > 0, e / (e + 1.0), 0.0)
        need = (e == 0) & dreamed
        if need.any():
            q[need] = self.dream_prior(S[need], T[need])
        return q

    def hop_beta(self, s, t):
        # Imagining the same move again is not new evidence, so a dreamed edge carries one prior (the learned
        # precision of similar dreams); real passive transitions accumulate as evidence.
        q = float(self._q([s], [t])[0])
        return self.kappa * q + self.ok.get((s, t), 0.0), self.kappa * (1 - q) + self.ko.get((s, t), 0.0)

    def _imagine_from(self, s, z):
        """One-step F3 imagination from the live latent: edges to known states, and new imagined states beyond."""
        if self.f3_latents is None:
            return
        for zn in self.f3_latents(np.asarray(z, np.float32)):
            j, d = self.nearest(zn)
            if j is not None and d <= self.radius:
                if j != s:
                    self._edge(s, j, dream=1.0); self.stats['imagined'] += 1
            elif self._new_this_tick < self.new_imag_per_tick:
                j = self._new_state(np.asarray(zn, np.float32), real=False)
                self._new_this_tick += 1
                self._edge(s, j, dream=1.0); self._edge(j, s, dream=0.5)

    def link_goal(self, zg, g):
        """Put the goal on the map: one F3 step imagined FROM the goal latent links it to its neighbourhood (both
        directions, depth 1; moves in a maze can be undone). Without this the goal only gets edges when the live
        latent happens to be one step from it, and risk has no route to work with."""
        if self.f3_latents is None or self.radius in (None, np.inf):
            return
        for zn in self.f3_latents(np.asarray(zg, np.float32)):
            j, d = self.nearest(zn)
            if j is None or d > self.radius:
                j = self._new_state(np.asarray(zn, np.float32), real=False)
            if j != g:
                self._edge(j, g, dream=1.0); self._edge(g, j, dream=1.0)

    def target_latent(self, z, t):
        """What the reflex is given for target state t. A real state: its real exemplar (H-JEPA reaches real
        encodings). An imagined state: a FRESH one-step F3 prediction from the live latent, under the macro-action
        whose prediction lands nearest t -- the same kind of subgoal H-JEPA's own level 3 hands level 2 (an F3 latent
        imagined from somewhere else is not reachable: 0/18 hops in the 24 Nov smoke)."""
        if self.real[t] or self.f3_latents is None or not self.fresh_f3_targets:
            return self.ex[t]
        Zn = self.f3_latents(np.asarray(z, np.float32))
        return Zn[int(np.linalg.norm(Zn - self.C[t][None], axis=1).argmin())]

    def infer_state(self, z, tick):
        s, d = self.observe(z, grow=tick)
        self.last_fit = round(d / self.radius, 2) if self.radius not in (None, np.inf) else -1.0
        if not tick:
            return s
        self._new_this_tick = 0
        if self.issued is not None:                                     # before the passive count below
            a, t = self.issued
            # made the hop: in t, or within one radius of t's centre (states overlap; nearest-centre alone is harsh)
            hit = s == t or (self.hit_radius and
                             float(np.linalg.norm(np.asarray(z, np.float32) - self.C[t])) <= self.radius)
            hit = hit or t in getattr(self, '_visited', ())          # passed through t at some frame (stream)
            if hit and s != t:
                s = t
            if hit:
                self.ok[(a, t)] += 1; self.stats['moves_ok'] += 1
            else:
                self.ko[(a, t)] += 1; self.stats['moves_failed'] += 1
            if self.e_real.get((a, t), 0.0) == 0 and self.ok[(a, t)] + self.ko[(a, t)] == 1:
                self.d_tests.append((a, t, float(hit), self.d_depth.get((a, t), 3)))   # first test of a dream-only hop
                self.d_try_all += 1; self.d_ok_all += hit
        if self.history and self.history[-1] != s and not getattr(self, '_stream', None):
            self._edge(self.history[-1], s, real=1.0)           # (with the stream, record() counts transitions)
        self._visited = set()
        self._imagine_from(s, z)
        return s

    # ------------------------------------------------------------------ dreaming aimed at the frontier
    def dream(self, seeds):
        if self.f3_batch is None or self.radius in (None, np.inf) or not len(seeds):
            return
        med_step = self.radius / self.rho
        Zs = np.repeat(np.asarray(seeds, np.float32), self.dream_n, axis=0)
        prev = [self._dream_assign(z) for z in Zs]
        alive = np.ones(len(Zs), bool)
        for depth in range(1, self.dream_h + 1):
            idx = np.flatnonzero(alive)
            if not len(idx):
                break
            Zn = self.f3_batch(Zs[idx])
            ok = np.linalg.norm(Zn - Zs[idx], axis=1) < 3 * med_step       # drift guard
            alive[idx[~ok]] = False
            for k, i in enumerate(idx):
                if ok[k]:
                    Zs[i] = Zn[k]
                    j = self._dream_assign(Zn[k])
                    if j != prev[i]:
                        self._edge(prev[i], j, dream=1.0, depth=depth)
                        self._edge(j, prev[i], dream=0.5, depth=depth)
                    prev[i] = j
                    self.stats['dream_steps'] += 1

    def frontier_score(self, s):
        """What there is to learn around s: information gain of its exits + a bonus for few confirmed exits."""
        exits = self.nbr.get(s, ())
        ig = sum(float(beta_bernoulli_ig(*self.hop_beta(s, t))) for t in exits)
        confirmed = sum(1 for t in exits if self.ok.get((s, t), 0) > 0 or self.e_real.get((s, t), 0) > 0)
        return ig + 1.0 / (1 + confirmed)

    def sleep(self, z, zg=None):
        real = [s for s in range(self.K) if self.real[s]]
        real.sort(key=self.frontier_score, reverse=True)
        seeds = [self.ex[s] for s in real[:self.frontier_seeds]] + [np.asarray(z, np.float32)]
        if zg is not None:
            seeds.append(np.asarray(zg, np.float32))
        self.dream(seeds)

    # ------------------------------------------------------------------ planning: one EFE objective
    def _edges(self):
        src, dst = [], []
        for s, ts in self.nbr.items():
            for t in ts:
                src.append(s); dst.append(t)
        src = np.asarray(src, np.int64); dst = np.asarray(dst, np.int64)
        if not len(src):
            return src, dst, np.zeros(0), np.zeros(0)
        q = self._q(src, dst)
        ok = np.array([self.ok.get((s, t), 0.0) for s, t in zip(src, dst)], np.float64)
        ko = np.array([self.ko.get((s, t), 0.0) for s, t in zip(src, dst)], np.float64)
        a, b = self.kappa * q + ok, self.kappa * (1 - q) + ko
        return src, dst, a / (a + b), beta_bernoulli_ig(a, b)

    def reach_prob(self, goal, src, dst, p, T):
        """V_{T-1} and V_T: best discounted probability of reaching a goal state within that many latent steps."""
        lam = self.time_discount
        V = np.zeros(self.K)
        V[list(goal)] = 1.0
        Vprev = V.copy()
        for _ in range(max(int(T), 1)):
            Vprev = V
            q = lam * (p * V[dst] + (1 - p) * V[src])
            Vn = lam * V
            np.maximum.at(Vn, src, q)
            Vn[list(goal)] = 1.0
            V = Vn
        return Vprev, V

    def info_value(self, src, dst, p, ig):
        U = np.zeros(self.K)
        for _ in range(self.u_horizon):
            val = ig + self.gamma * (p * U[dst] + (1 - p) * U[src])
            Un = np.zeros(self.K)
            np.maximum.at(Un, src, val)
            U = Un
        return U

    def decide(self, s, goal):
        src, dst, p, ig = self._edges()
        here = np.flatnonzero(src == s)
        if not len(here):
            return None, 'none'
        U = self.info_value(src, dst, p, ig)
        E = ig[here] + self.gamma * (p[here] * U[dst[here]] + (1 - p[here]) * U[s])
        mode = 'epistemic'
        score = E.copy()
        if goal:
            Vprev, _ = self.reach_prob(goal, src, dst, p, self.ticks_left)
            q = p[here] * Vprev[dst[here]] + (1 - p[here]) * Vprev[s]
            if q.max() <= 0:                                   # a route exists only beyond the time left: still go
                Vprev, _ = self.reach_prob(goal, src, dst, p, 4 * self.u_horizon)
                q = p[here] * Vprev[dst[here]] + (1 - p[here]) * Vprev[s]
            if q.max() > 0:
                mode = 'pragmatic'
                score = np.log(q + 1e-12) + self.eta * ig[here]
        if mode == 'epistemic' and E.max() < self.ig_eps:
            return None, 'none'
        if self.last_target is not None:
            score = score + self.habit * (dst[here] == self.last_target)
        return int(dst[here[int(score.argmax())]]), mode

    def _random_step(self, z):
        """Random F3 step from the live latent, as an imagined state to hand down."""
        Zn = self.f3_latents(np.asarray(z, np.float32))
        zn = Zn[self.rng.integers(len(Zn))]
        j, d = self.nearest(zn)
        if j is None or d > self.radius:
            j = self._new_state(np.asarray(zn, np.float32), real=False)
        return j

    def step(self, z, zg, observe=True, lr=1.0, tick=True):
        z = np.asarray(z, np.float32)
        self._calibrate(z)
        goal = set() if self.explore else self.goal_states(zg)
        if self.sleep_pending:
            self.observe(z)
            if self.dreaming and self.f3_batch is not None:
                self.sleep(z, None if self.explore else zg)
            self.sleep_pending = False
        if goal and tick and len(self.history) % 4 == 0:            # task start, then every 4 latent steps
            self.link_goal(zg, next(iter(goal)))
        s = self.infer_state(z, tick)
        if tick:
            self.history.append(s)
        info = dict(s=s, K=self.K)
        if s in goal:
            self.stats['at_goal'] += 1
            self.issued, self.last_target = None, None
            return None, info | dict(reason='at_goal')
        if self.explore and self.explorer == 'random':
            target, mode = self._random_step(z), 'random_step'
        else:
            target, mode = self.decide(s, goal)
            if target is None and self.explore and self.f3_latents is not None:
                target, mode = self._random_step(z), 'random_step'
        if target is None or target == s:
            self.stats['handover'] += 1
            self.issued, self.last_target = None, None
            return None, info | dict(reason='no_route')
        self.stats[mode] += 1
        self.issued, self.last_target = (s, int(target)), int(target)
        return int(target), info | dict(target=int(target), mode=mode)

    # ------------------------------------------------------------------ reporting
    def map_summary(self):
        src, dst, p, ig = self._edges()
        tried = np.array([self.ok.get((s, t), 0) + self.ko.get((s, t), 0) > 0 for s, t in zip(src, dst)], bool)
        return dict(states=self.K, real_states=int(sum(self.real)), edges=int(len(src)), tried=int(tried.sum()),
                    confirmed=int(sum(self.ok.get((s, t), 0) > 0 for s, t in zip(src, dst))),
                    refuted=int(((p < 0.2) & tried).sum()), mean_ig=float(ig.mean()) if len(ig) else 0.0)
