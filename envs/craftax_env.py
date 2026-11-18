"""Craftax-Classic (JAX re-implementation of Crafter) with pixels rendered exactly like CrafterPlay videos.

Frames: renderer at 16 px/block -> 144x144 RGB (same as the dataset videos), then INTER_AREA resize to 64.
"""
from __future__ import annotations

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import Achievement
from craftax.craftax_classic.envs.craftax_symbolic_env import \
    CraftaxClassicSymbolicEnvNoAutoReset as CraftaxClassicSymbolicEnv   # keep final state (achievements) at death
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

ACHIEVEMENTS = [a.name.lower() for a in Achievement]


class Craftax:
    def __init__(self, seed: int = 0, size: int = 64):
        self.env = CraftaxClassicSymbolicEnv()
        self.params = self.env.default_params
        self.size = size
        self._step = jax.jit(self.env.step)
        self._reset = jax.jit(self.env.reset)
        self._render = jax.jit(make_craftax_pixel_renderer(16))
        self.key = jax.random.PRNGKey(seed)

    def reset(self):
        self.key, k = jax.random.split(self.key)
        _, self.state = self._reset(k, self.params)
        return self.observe()

    def step(self, action: int):
        self.key, k = jax.random.split(self.key)
        _, self.state, reward, done, info = self._step(k, self.state, int(action), self.params)
        return self.observe(), float(reward), bool(done)

    def frame144(self) -> np.ndarray:
        return np.asarray(self._render(self.state)).clip(0, 255).astype(np.uint8)

    def observe(self) -> dict:
        f = self.frame144()
        interp = cv2.INTER_NEAREST if self.size > 144 else cv2.INTER_AREA   # match training-time resizing
        return {'pixels': cv2.resize(f, (self.size, self.size), interpolation=interp),
                'frame144': f,
                'pos': np.asarray(self.state.player_position).copy(),
                'achievements': np.asarray(self.state.achievements).copy(),
                'map': np.asarray(self.state.map)}


def crafter_score(success_rates_pct: np.ndarray) -> float:
    """Hafner (2021): geometric mean of per-achievement success rates (in %)."""
    return float(np.exp(np.mean(np.log(1 + np.asarray(success_rates_pct)))) - 1)
