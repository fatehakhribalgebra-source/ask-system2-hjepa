"""A renormalising generative model (RGM; Friston et al. 2025, "From pixels to planning") over H-JEPA level 3,
used as H-JEPA's LEVEL-3 ACTION PLANNER (design A): it returns level-3 macro-actions, exactly where the Tschantz
planner sits.

Faithful to SPM (spm_faster_structure_learning / spm_induction, read from the MATLAB sources in atomresearch/RGMs)
in the parts that define the method; minimal elsewhere.

Level 1 (states s, paths u)
  s : discrete state of z3 = nearest of K centroids in the whitened 32-d view of z3 (the state VAE whitener).
      z3 has no spatial layout, so RGM's spatial blocking reduces to a single group; the SVD-and-quantise step is
      replaced by k-means on the whitened (PCA) coordinates, which is the same reduction with a learned codebook.
  u : path = one of the 12 real macro-action prototypes (on level 3's LayerNorm manifold), so every action RGM
      can choose is a valid level-3 move.
  B1[s' | s, u] : Dirichlet counts of observed transitions (structure learning by counting unique transitions).
Level 2 (temporal renormalisation, T = 2)
  e : event = (initial state s_t, path word (u_t, u_t+1)); D2(e) = s_t, E2(e) = (u_t, u_t+1). One level-2 step is a
      word of two level-3 macro-actions (40 env steps).
  B2[e' | e] : counts of consecutive events in the real data.
Planning = inductive inference (spm_induction): transitions are thresholded at u = 1/16 of each column's maximum,
goal states are propagated BACKWARDS with logical operators for at most 16 steps, and the agent takes the action
that best realises the intended next state (control as inference: maximise the predicted probability of the
intended state, the accuracy term of free energy). We keep the whole layer structure (distance-to-goal per state)
instead of only the next layer, so a committed route can be read off the same computation.
"""
from __future__ import annotations

import numpy as np
import torch

U_THRESH = 1 / 16          # spm_induction: probability threshold relative to the column maximum
MAX_DEPTH = 16             # spm_induction: min(N, 16)


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C


def backward_layers(adj: np.ndarray, goal: np.ndarray, depth: int = MAX_DEPTH) -> np.ndarray:
    """adj[i, j] = True if j -> i is an allowed transition (column = from, as in SPM's B{f}(:, from)).
    Returns dist[j] = smallest n <= depth with a path j -> goal of n steps (0 = goal), or -1 if none.
    This is spm_induction's backward protocol I(:, n+1) = any(B(I(:, n), :), 1), kept for every layer."""
    dist = np.full(adj.shape[1], -1, int)
    layer = goal.copy()
    dist[layer] = 0
    for n in range(1, depth + 1):
        layer = adj[layer].any(0)                  # states with an allowed transition INTO the previous layer
        new = layer & (dist < 0)
        dist[new] = n
        if not new.any():
            break
    return dist


class RGMLevel3:
    def __init__(self, centroids, protos, B1, events, B2, tau, levels=2, surprise_thresh=None):
        self.C, self.P = centroids, protos                       # (K, d) whitened centroids, (12, 4) prototypes
        self.B1, self.events, self.B2 = B1.copy(), events, B2    # (K, K, U) counts; (E, 3) [s, u0, u1]; (E, E)
        self.tau, self.levels = tau, levels
        self.surprise_thresh = surprise_thresh                   # 90th pct of surprise on held-out real transitions
        self.K, self.U = B1.shape[0], B1.shape[2]
        if B2 is not None and len(B2):
            c2 = B2 / np.maximum(B2.sum(0, keepdims=True), 1e-12)
            self.adj2_full = (c2 > c2.max(0, keepdims=True) * U_THRESH) & (B2 > 0)
        else:
            self.adj2_full = None
        self.blocked: set[tuple[int, int]] = set()
        self._refresh()

    def _refresh(self):
        """Rebuild the EFFECTIVE transition model from the counts B1 and the blocked moves (state, action):
        b1 = p(s' | s, u) with blocked columns removed, adj1 = allowed s -> s', adj2 = allowed event transitions."""
        b = self.B1 / np.maximum(self.B1.sum(0, keepdims=True), 1e-12)   # columns = p(s' | s, u)
        for s, u in self.blocked:
            b[:, s, u] = 0.0
        self.b1 = b
        thr = (b > b.max(0, keepdims=True) * U_THRESH) & (b > 0)        # allowed s -> s' under some path u
        self.adj1 = thr.any(2)                                            # adj1[s', s]
        if self.adj2_full is not None:
            ok = np.ones(len(self.events), bool)
            for s, u in self.blocked:
                ok &= ~((self.events[:, 0] == s) & (self.events[:, 1] == u))
            self.adj2 = self.adj2_full & ok[None, :] & ok[:, None] if self.blocked else self.adj2_full
            self.ev_ok = ok
        else:
            self.adj2, self.ev_ok = None, None
        self._reach_cache = (None, None)

    def set_blocked(self, blocked):
        blocked = set(blocked)
        if blocked != self.blocked:
            self.blocked = blocked
            self._refresh()

    def learn(self, q_from: np.ndarray, u: int, q_to: np.ndarray, lr: float = 1.0):
        """Continual learning from a REAL transition: add soft Dirichlet counts B1[s', s, u] += lr q_from(s) q_to(s')."""
        self.B1[:, :, int(u)] += lr * np.outer(q_to, q_from)
        self._refresh()

    # ------------------------------------------------------------------ fitting (structure learning by counting)
    @classmethod
    def fit(cls, S, A, EP, protos, k=64, levels=2, seed=0):
        """S: (N, d) whitened z3 at level-3 steps, in time order within each episode; A: (N, 4) true macro-actions;
        EP: (N,) episode ids. Consecutive rows of one episode are consecutive level-3 steps."""
        C = kmeans(S, k, seed=seed)
        s = ((S[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        u = ((A[:, None] - protos[None]) ** 2).sum(-1).argmin(1)
        tau = float(((S - C[s]) ** 2).sum(-1).mean())
        U = len(protos)
        B1 = np.zeros((k, k, U))
        ev_index, ev_list, pairs = {}, [], []
        for e in np.unique(EP):
            idx = np.flatnonzero(EP == e)
            se, ue = s[idx], u[idx]
            for t in range(len(idx) - 1):
                B1[se[t + 1], se[t], ue[t]] += 1
            seq = []
            for t in range(len(idx) - 2):                     # both phases of the T = 2 renormalisation
                key = (se[t], ue[t], ue[t + 1])
                if key not in ev_index:
                    ev_index[key] = len(ev_list); ev_list.append(key)
                seq.append(ev_index[key])
            for t in range(len(seq) - 2):
                pairs.append((seq[t + 2], seq[t]))            # event t (steps t, t+1) -> event t+2
        E = len(ev_list)
        B2 = np.zeros((E, E))
        for to, fr in pairs:
            B2[to, fr] += 1
        return cls(C, protos, B1, np.asarray(ev_list), B2 if levels >= 2 else None, tau, levels)

    # ------------------------------------------------------------------ inference
    def posterior(self, y: np.ndarray) -> np.ndarray:
        """Q(s | z3): softmax of -||y - c_s||^2 / tau (tau = mean within-state squared distance)."""
        d2 = ((self.C - y[None]) ** 2).sum(-1)
        q = np.exp(-(d2 - d2.min()) / self.tau)
        return q / q.sum()

    def goal_states(self, yg: np.ndarray, mass: float = 0.5) -> np.ndarray:
        """Intended states (id.hid): the most probable states of the goal latent covering `mass` of its posterior."""
        q = self.posterior(yg)
        order = np.argsort(-q)
        keep = order[: int(np.searchsorted(np.cumsum(q[order]), mass)) + 1]
        g = np.zeros(self.K, bool); g[keep] = True
        return g

    # ------------------------------------------------------------------ planning by induction
    def plan(self, y: np.ndarray, yg: np.ndarray, H: int) -> tuple[np.ndarray | None, dict]:
        """Return H macro-action prototype indices (or None if the agent is already in a goal state / no path)."""
        goal = self.goal_states(yg)
        q = self.posterior(y)
        s_now = int(q.argmax())
        if goal[s_now]:
            return None, dict(reason='at_goal', s=s_now)
        dist1 = backward_layers(self.adj1, goal)
        if self.levels >= 2 and self.adj2 is not None:
            acts, info = self._plan_level2(q, goal, dist1, H)
            if acts is not None:
                return acts, info | dict(s=s_now, d1=int(dist1[s_now]))
        acts, info = self._plan_level1(q, dist1, H)
        return acts, info | dict(s=s_now, d1=int(dist1[s_now]))

    def _act_level1(self, q, dist1):
        """Intended next states R = layer (n - 1) where n is the first layer holding posterior mass > u (as in
        spm_induction); action u* = argmax_u sum_s q(s) sum_{s' in R} p(s' | s, u)."""
        on = dist1 >= 0
        if not (q[on] > 0).any():
            return None, None
        layers = sorted(set(dist1[on].tolist()) - {0}) or [0]
        n = next((L for L in layers if q[dist1 == L].sum() > U_THRESH), None)
        if n is None:
            return None, None
        R = dist1 == (n - 1)
        score = np.einsum('s,tsu->u', q, self.b1 * R[:, None, None])
        return int(score.argmax()), n

    def _plan_level1(self, q, dist1, H):
        acts, qq = [], q.copy()
        n0 = None
        goal = dist1 == 0
        for _ in range(H):
            if acts and qq[goal].sum() > 0.5:                  # route ends once the goal is the likely state
                break
            a, n = self._act_level1(qq, dist1)
            if a is None:
                break
            n0 = n0 if n0 is not None else n
            acts.append(a)
            qq = self.b1[:, :, a] @ qq                     # predicted state distribution under the chosen path
            qq = qq / max(qq.sum(), 1e-12)
        if not acts:
            return None, dict(reason='no_path_l1', level=1)
        n_real = len(acts)
        acts += [acts[-1]] * (H - len(acts))
        return np.asarray(acts[:H]), dict(level=1, n=n0, plan_len=n_real)

    def _plan_level2(self, q, goal, dist1, H):
        """Level-2 induction over events (state + two-action word). Goal events: events starting in a goal state
        or whose word leads (under B1) into one. The current event is chosen among events starting at the current
        state on the shortest event path; its word gives the next two macro-actions, then successors follow."""
        ev = self.events
        start_goal = goal[ev[:, 0]]
        reach_p = self._reach_p(goal)
        dist2 = backward_layers(self.adj2, start_goal | (reach_p > U_THRESH))
        q_ev = q[ev[:, 0]]
        cand = (q_ev > U_THRESH * q_ev.max()) & (dist2 >= 0) & self.ev_ok
        if not cand.any():
            return None, dict(reason='no_path_l2')
        best = np.flatnonzero(cand)
        dmin = dist2[best].min()
        best = best[dist2[best] == dmin]
        e = int(best[np.argmax(q_ev[best] * self._event_score(best, dmin, reach_p))])
        acts = [int(ev[e, 1]), int(ev[e, 2])]
        while len(acts) < H and dist2[e] > 0:                   # follow shortest successors, stop at the goal
            succ = np.flatnonzero(self.adj2[:, e] & (dist2 >= 0) & (dist2 < dist2[e]))
            if not len(succ):
                break
            d = dist2[succ].min()
            succ = succ[dist2[succ] == d]
            e = int(succ[np.argmax(self.B2[succ, e] * self._event_score(succ, d, reach_p))])
            acts += [int(ev[e, 1]), int(ev[e, 2])]
        n_real = len(acts)
        acts += [acts[-1]] * (H - len(acts))
        return np.asarray(acts[:H]), dict(level=2, n=int(dmin), plan_len=n_real)

    def _reach_p(self, goal):
        """p(goal state reached by the end of each event's two-action word | event's start state), under B1."""
        if getattr(self, '_reach_cache', (None,))[0] is not None and np.array_equal(self._reach_cache[0], goal):
            return self._reach_cache[1]
        ev = self.events
        p1 = self.b1[:, ev[:, 0], ev[:, 1]]                     # (K, E) after first action
        p2 = np.einsum('tsE,sE->tE', self.b1[:, :, ev[:, 2]], p1)  # (K, E) after second action
        reach = p2[goal].sum(0) + p1[goal].sum(0) * 0.5         # count passing through the goal half
        self._reach_cache = (goal.copy(), reach)
        return reach

    def _event_score(self, idx, d, reach_p):
        """Tie-break among equally short events: at the goal layer, the event most likely to land in the goal;
        otherwise the most experienced event (largest Dirichlet count)."""
        if d == 0:
            return reach_p[idx] + 1e-6
        return self.B2[:, idx].sum(0) + 1e-3

    # ------------------------------------------------------------------ commit or replan
    def route_success(self, q: np.ndarray, acts, goal: np.ndarray) -> float:
        """P(goal reached at some point while executing `acts` from belief q), under B1, goal absorbing."""
        q = q.copy()
        absorbed = float(q[goal].sum()); q[goal] = 0.0
        for u in acts:
            q = self.b1[:, :, int(u)] @ q
            absorbed += float(q[goal].sum()); q[goal] = 0.0
        return min(1.0, absorbed)

    def extend_route(self, q, acts, goal, H_max, p_target=0.5):
        """Lengthen a route with level-1 induction steps (from the predicted belief) until it reaches the goal with
        probability >= p_target, or H_max actions. Induction's 1/16 threshold only says the goal is POSSIBLE at the
        end of a route; a committed route should make it LIKELY."""
        dist1 = backward_layers(self.adj1, goal)
        acts = [int(u) for u in acts]
        qq = q.copy()
        for u in acts:
            qq = self.b1[:, :, u] @ qq; qq = qq / max(qq.sum(), 1e-12)
        while len(acts) < H_max and self.route_success(q, acts, goal) < p_target:
            u, _ = self._act_level1(qq, dist1)
            if u is None:
                break
            acts.append(u)
            qq = self.b1[:, :, u] @ qq; qq = qq / max(qq.sum(), 1e-12)
        return acts

    def commit_or_replan(self, y, yg, remaining, H_max, switch_cost=0.5, p_target=0.5):
        """Expected-free-energy choice between two policies: continue the committed route (`remaining` actions) or
        switch to a freshly planned one. G = risk = -ln P(goal | policy); switching pays a habit cost (nats) that
        favours continuing. Returns (replan?, fresh acts, fresh info, terms)."""
        goal = self.goal_states(yg)
        q = self.posterior(y)
        fresh, info = self.plan(y, yg, H_max)
        if fresh is None:
            return True, None, info, {}
        fresh = fresh[: max(1, int(info.get('plan_len', len(fresh))))]
        fresh = self.extend_route(q, fresh, goal, H_max, p_target)
        eps = 1e-6
        p_cont = self.route_success(q, remaining, goal) if len(remaining) else 0.0
        p_new = self.route_success(q, fresh, goal)
        G_cont, G_new = -np.log(p_cont + eps), -np.log(p_new + eps) + switch_cost
        return bool(G_new < G_cont), fresh, info, dict(p_continue=p_cont, p_replan=p_new,
                                                       G_continue=float(G_cont), G_replan=float(G_new))

    # ------------------------------------------------------------------ bridge: closed loop + two experts
    def bridge_decide(self, q, yg, f3_post, w_f3=1.0, H=5):
        """One decision from the REAL belief q (closed loop). Progress set R = states strictly closer to the goal
        (backward induction layers, goal included). Two experts predict, for each candidate action u, the probability
        that the next level-3 step makes progress:
            RGM : P_rgm(R | q, u) = sum_s q(s) sum_{s' in R} b1[s', s, u]
            F3  : P_f3(R | z, u)  = posterior of H-JEPA's one-step imagined z3 under u, mass on R
        All 12 actions are scored, so either expert can propose a move the other rates low; the winner maximises
        the product of experts  ln P_rgm + w_f3 ln P_f3,  w_f3 = F3's precision relative to RGM's (learned from
        reality).
        f3_post(list of actions) -> list of posteriors over states. Returns (H actions, predictions, info) or
        (None, None, info) when the agent is in a goal state."""
        goal = self.goal_states(yg)
        s_now = int(q.argmax())
        if goal[s_now]:
            return None, None, dict(reason='at_goal', s=s_now)
        dist1 = backward_layers(self.adj1, goal)
        on = dist1 >= 0
        d_now = float(q[on] @ dist1[on] / max(q[on].sum(), 1e-12))  # expected distance over reachable states
        d_ref = int(dist1[s_now]) if dist1[s_now] >= 0 else int(np.ceil(d_now))
        R = on & (dist1 < d_ref)                                    # strictly closer than the current layer
        if not R.any():
            R = goal.copy()
        p_rgm = np.einsum('s,tsu->u', q, self.b1 * R[:, None, None])
        if p_rgm.max() <= 0:                                       # no admissible move under the model
            return None, None, dict(reason='no_progress', s=s_now)
        cand = np.arange(self.U)                                     # full product of experts over all actions
        posts = f3_post([int(u) for u in cand]) if w_f3 > 0 else [None] * len(cand)
        p_f3 = np.array([float(p[R].sum()) if p is not None else 1.0 for p in posts])
        score = np.log(p_rgm[cand] + 1e-6) + w_f3 * np.log(p_f3 + 1e-6)
        i = int(score.argmax())
        u = int(cand[i])
        q_next = self.b1[:, :, u] @ q; q_next = q_next / max(q_next.sum(), 1e-12)
        rest, _ = self._plan_level1(q_next, dist1, max(1, H - 1))
        acts = [u] + ([int(x) for x in rest] if rest is not None else [u] * (H - 1))
        preds = dict(rgm=q_next, f3=posts[i])
        n_adm = int((p_rgm >= U_THRESH * p_rgm.max()).sum())        # actions induction alone would admit
        return np.asarray(acts[:H]), preds, dict(s=s_now, d_now=round(d_now, 2), n_cand=n_adm, u=u,
                                                 p_rgm=round(float(p_rgm[u]), 3), p_f3=round(float(p_f3[i]), 3),
                                                 w_f3=round(float(w_f3), 3))

    # ------------------------------------------------------------------ feedback from two realities
    @staticmethod
    def surprise(expected: np.ndarray, observed: np.ndarray) -> float:
        """-ln sum_s expected(s) observed(s): how unlikely the observed state is under the predicted belief."""
        return float(-np.log(float(expected @ observed) + 1e-12))

    def expected_beliefs(self, q: np.ndarray, acts) -> list[np.ndarray]:
        """RGM's own predicted belief after each action of a route (effective b1)."""
        out = []
        for u in acts:
            q = self.b1[:, :, int(u)] @ q; q = q / max(q.sum(), 1e-12); out.append(q)
        return out

    def verified_route(self, y, yg, H_max, imagine, tries=3, p_target=0.5):
        """INNER LOOP (imagined reality). Plan a route to the goal, let H-JEPA's level-3 predictor F3 imagine it
        (`imagine(acts) -> list of whitened z3, one per action`), and compare every imagined state with RGM's expected
        belief. If a step is too surprising, block that (state, action) move and replan; up to `tries` attempts.
        Commit the route with the fewest surprising steps. Its expected states are the F3-CHECKED beliefs (posterior
        of the imagined z3), which the outer loop compares with reality.
        Returns (acts, expected beliefs, info) or (None, None, info) at the goal / without a path."""
        goal = self.goal_states(yg)
        q = self.posterior(y)
        blocked, best = set(), None
        log = []
        try:
            for attempt in range(tries):
                self.set_blocked(blocked)
                acts, info = self.plan(y, yg, H_max)
                if acts is None:
                    if best is None:
                        return None, None, info | dict(attempts=attempt)
                    break
                acts = self.extend_route(q, acts[: max(1, int(info.get('plan_len', len(acts))))], goal, H_max, p_target)
                exp = self.expected_beliefs(q, acts)
                imag = [self.posterior(yi) for yi in imagine(acts)]
                sur = [self.surprise(e, o) for e, o in zip(exp, imag)]
                bad = [k for k, v in enumerate(sur) if v > self.surprise_thresh]
                log.append(dict(len=len(acts), bad=len(bad), first_bad=bad[0] if bad else None,
                                max_surprise=float(max(sur)) if sur else 0.0))
                cand = (len(bad), len(acts))
                if best is None or cand < best[0]:
                    best = (cand, acts, imag, info)
                if not bad:
                    break
                k = bad[0]
                s_from = int(q.argmax()) if k == 0 else int(imag[k - 1].argmax())
                blocked = blocked | {(s_from, int(acts[k]))}
        finally:
            self.set_blocked(set())
        _, acts, imag, info = best
        return acts, imag, info | dict(attempts=len(log), inner=log, blocked=len(blocked))

    # ------------------------------------------------------------------ io
    def state_dict(self):
        return dict(C=self.C, P=self.P, B1=self.B1, events=self.events, B2=self.B2, tau=self.tau, levels=self.levels,
                    surprise_thresh=self.surprise_thresh, calib=getattr(self, 'calib', {}))

    @classmethod
    def from_state(cls, d, levels=None):
        m = cls(d['C'], d['P'], d['B1'], d['events'], d['B2'], d['tau'], d['levels'] if levels is None else levels,
                d.get('surprise_thresh'))
        m.calib = d.get('calib', {})
        return m


def protos_tensor(rgm: RGMLevel3, idx, device='cuda') -> torch.Tensor:
    return torch.as_tensor(rgm.P[np.asarray(idx)], dtype=torch.float32, device=device)
