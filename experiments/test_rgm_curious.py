"""Toy check of CuriousRGM, numpy only (no H-JEPA, no simulator).

World: the square [0, 5]^2 with a wall at x = 2.5 for y < 4 (gap at the top). The "latent" is the 2-D position.
Imagination (stand-in for F3) steps 0.5 in 12 directions and IGNORES walls, so dreams cross the wall. The reflex
(stand-in for H-JEPA's lower levels) moves at most 0.5 toward the handed-down exemplar and is stopped by the wall.
Start (0.5, 0.5), goal (4.5, 0.5): the straight line is blocked; the route goes round through the gap.

Checks: (1) exploring builds a map (coverage) and refutes dreamed edges through the wall; curious vs random;
(2) after exploring, the goal is reached through the gap within the time budget.

  python experiments/test_rgm_curious.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from system2.rgm_calib import CalibRGM  # noqa: E402
from system2.rgm_curious import CuriousRGM  # noqa: E402

CALIB = '--calib' in sys.argv   # CalibRGM: the toy imagination judges each hop, reality calibrates it

STEP = 0.5
DIRS = np.stack([np.cos(np.linspace(0, 2 * np.pi, 12, endpoint=False)),
                 np.sin(np.linspace(0, 2 * np.pi, 12, endpoint=False))], 1).astype(np.float32)


def blocked(p, q):
    if (p[0] - 2.5) * (q[0] - 2.5) >= 0:
        return False
    y = p[1] + (q[1] - p[1]) * (2.5 - p[0]) / (q[0] - p[0])
    return y < 4.0


def reflex(p, target):
    d = target - p
    n = np.linalg.norm(d)
    q = p + (d if n <= STEP else d / n * STEP)
    q = np.clip(q, 0.05, 4.95).astype(np.float32)
    return p if blocked(p, q) else q


def imagine(Z, steps, g, halluc):
    """Imagined next positions: a learned model usually respects walls; with probability `halluc` it does not."""
    out = np.clip(Z + steps, 0.05, 4.95).astype(np.float32)
    for i in range(len(out)):
        if blocked(Z[i], out[i]) and g.random() >= halluc:
            out[i] = Z[i]
    return out


def make(explorer='curious', dream=True, seed=0, halluc=0.25):
    m = CalibRGM() if CALIB else CuriousRGM()
    m.explorer, m.rng = explorer, np.random.default_rng(seed)
    g = np.random.default_rng(seed + 1)
    m.f3_latents = lambda z: imagine(np.repeat(z[None], len(DIRS), 0), STEP * DIRS, g, halluc)

    def f3_batch(Z):
        a = g.uniform(0, 2 * np.pi, len(Z))
        return imagine(Z, STEP * np.stack([np.cos(a), np.sin(a)], 1), g, halluc)
    m.f3_batch, m.dreaming = f3_batch, dream
    if CALIB:
        m.f3_reach = lambda Zs, Zt: np.array([np.linalg.norm(imagine(np.repeat(zs[None], len(DIRS), 0), STEP * DIRS, g, halluc)
                                                             - zt, axis=1).min() for zs, zt in zip(Zs, Zt)])
    return m


def episode(m, start, goal, ticks, explore):
    m.reset(); m.explore = explore
    p = np.asarray(start, np.float32)
    path = [p]
    for k in range(ticks):
        m.ticks_left = ticks - k
        tgt, info = m.step(p, np.asarray(goal, np.float32))
        if tgt is None:
            if info.get('reason') == 'at_goal':
                return True, k, path
            continue
        p = reflex(p, m.target_latent(p, tgt))
        path.append(p)
    _, d = m.nearest(p)
    return (not explore) and np.linalg.norm(p - goal) < m.radius, ticks, path


def coverage(m):
    real = np.stack([m.ex[s] for s in range(m.K) if m.real[s]])
    return len({tuple(c) for c in np.floor(real).astype(int).tolist()}) / 25


def wall_edges(m):
    """Dreamed edges that cross the wall, and how many of the TRIED ones ended refuted (p < 0.2)."""
    cross = tried = refuted = 0
    for s, ts in m.nbr.items():
        for t in ts:
            if blocked(m.ex[s], m.ex[t]):
                cross += 1
                if m.ok.get((s, t), 0) + m.ko.get((s, t), 0) > 0:
                    tried += 1
                    a, b = m.hop_beta(s, t)
                    refuted += a / (a + b) < 0.35
    return cross, tried, refuted


def main():
    start, goal = (0.5, 0.5), (4.5, 0.5)
    rng = np.random.default_rng(7)
    starts = [start] + [tuple(rng.uniform(0.3, 4.7, 2)) for _ in range(9)]
    print('| imagination through walls | explorer | dream | coverage after 10 x 17 steps | states (real) | '
          'wall edges tried / refuted | goal solved | steps |')
    print('|---|---|---|---|---|---|---|---|')
    out = {}
    for halluc in (0.25, 1.0):
        for explorer in ('curious', 'random'):
            for dream in (True, False):
                sol, steps, cov = [], [], []
                for seed in range(5):
                    m = make(explorer, dream, seed, halluc)
                    for s0 in starts:
                        episode(m, s0, goal, 17, explore=True)
                    cov.append(coverage(m))
                    cross, tried, ref = wall_edges(m)
                    ok, k, _ = episode(m, start, goal, 40, explore=False)
                    sol.append(ok); steps.append(k)
                out[(halluc, explorer, dream)] = np.mean(sol)
                print(f'| {halluc:.0%} | {explorer} | {dream} | {np.mean(cov):.2f} | {m.K} ({sum(m.real)}) | {tried} / '
                      f'{ref} (of {cross}) | {sum(sol)}/5 | {np.mean([k for k, o in zip(steps, sol) if o] or [0]):.1f} |')
    # risk sanity: probable beats short. Two routes s0 -> goal: 1 hop with p = 0.2, or 2 hops with p = 0.9 each.
    m = CuriousRGM(); m.radius, m.tau = 1.0, 1.0
    for z in ([0, 0], [5, 0], [2, 3], [2.5, 6]):
        m._new_state(np.asarray(z, np.float32))
    m._edge(0, 1, real=1.0); m.ko[(0, 1)] = 8                       # short but doubtful
    for e in ((0, 2), (2, 1)):
        m._edge(*e, real=20.0); m.ok[e] = 20
    m.e_real[(1, 3)] = 0                                            # (state 3: unrelated)
    m.ticks_left = 6
    t, mode = m.decide(0, {1})
    m.ticks_left = 1
    t1, _ = m.decide(0, {1})
    print(f'\nrisk check: 6 steps left -> hop to {t} ({mode}; 2 = the probable 2-hop route), '
          f'1 step left -> hop to {t1} (1 = the only route that can still make it)')
    assert t == 2 and t1 == 1, 'risk planner does not trade length against probability'
    # What holds (24 Nov 2026): curiosity + one-step imagination finds the way round the wall, random exploration does
    # not. Known failure: with sleep-dreaming (F3 rollouts) the goal phase still spends its time on dreamed shortcuts
    # through the wall -- left visible in the table, not asserted.
    assert out[(0.25, 'curious', False)] >= 0.6, 'curious explorer should reach the goal round the wall'
    assert out[(0.25, 'curious', False)] > out[(0.25, 'random', False)], 'curiosity should beat random exploration'
    print('OK (dream rows: known failure, see table)')


if __name__ == '__main__':
    main()
