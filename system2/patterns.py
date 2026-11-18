"""Pattern sources ("Platonic seeds") and their structure-destroying controls.

Every source is a PatternStream: reset(offset) then next() -> np.ndarray of shape (dim,).
Image-like patterns (fractal, art, cellular automata) are read in raster order, one
label per step, as in Ertle, Levin & Scheutz (2025); the emitted vector is a one-hot
window of the last `window` labels, so the reservoir sees local structure.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- generators
def halley_fractal(a: int = 3, b: int = 1, res: int = 128, bounds=((-2, 2), (-2, 2)),
                   max_it: int = 40, eps: float = 1e-6, n_colors: int = 3) -> np.ndarray:
    """Labelled Halley-method fractal of f(x) = x(x^a - b) (Free Lunch, Alg. 2).

    Returns an int array (res, res) with labels in [0, n_colors]; label n_colors = no convergence,
    so the default gives 4 labels, matching Free Lunch's 4-action label space.
    """
    xs = np.linspace(*bounds[0], res)
    ys = np.linspace(*bounds[1], res)
    z = xs[None, :] + 1j * ys[:, None]
    labels = np.full(z.shape, n_colors, dtype=np.int64)
    done = np.zeros(z.shape, dtype=bool)
    g_prev = np.ones_like(z)
    for k in range(max_it):
        f = z * (z ** a - b)                      # x^(a+1) - b x
        df = (a + 1) * z ** a - b
        d2f = (a + 1) * a * z ** (a - 1)
        denom = 2 * df ** 2 - f * d2f
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        z = z - 2 * f * df / denom
        conv = (~done) & (np.abs(z - g_prev) < eps)
        if k % 2 == 0:
            labels[conv] = k % n_colors
            done |= conv
        g_prev = z
    return labels


def halley_basins(a: int = 3, b: int = 1, res: int = 128, bounds=((-2, 2), (-2, 2)),
                  max_it: int = 60, n_labels: int = 4) -> np.ndarray:
    """Halley fractal coloured by *which root* each point converges to (basins of attraction).

    Roots of x(x^a - b) are 0 and the a-th roots of b; label = root index mod n_labels.
    Balanced label histogram, fractal basin boundaries.
    """
    xs = np.linspace(*bounds[0], res)
    ys = np.linspace(*bounds[1], res)
    z = xs[None, :] + 1j * ys[:, None]
    for _ in range(max_it):
        f = z * (z ** a - b)
        df = (a + 1) * z ** a - b
        d2f = (a + 1) * a * z ** (a - 1)
        denom = 2 * df ** 2 - f * d2f
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        z = z - 2 * f * df / denom
    roots = np.concatenate([[0], b ** (1 / a) * np.exp(2j * np.pi * np.arange(a) / a)])
    idx = np.argmin(np.abs(z[..., None] - roots), axis=-1)
    return (idx % n_labels).astype(np.int64)


def rule30(width: int = 128, steps: int = 128, seed: int = 0) -> np.ndarray:
    row = np.zeros(width, dtype=np.int64)
    row[width // 2] = 1
    out = [row]
    for _ in range(steps - 1):
        l, r = np.roll(row, 1), np.roll(row, -1)
        row = l ^ (row | r)
        out.append(row)
    return np.stack(out)


def game_of_life(size: int = 64, steps: int = 64, density: float = 0.3, seed: int = 0) -> np.ndarray:
    """Space-time slice: rows are successive generations of the middle row band."""
    rng = np.random.default_rng(seed)
    g = (rng.random((size, size)) < density).astype(np.int64)
    frames = []
    for _ in range(steps):
        n = sum(np.roll(np.roll(g, i, 0), j, 1) for i in (-1, 0, 1) for j in (-1, 0, 1)) - g
        g = ((n == 3) | ((g == 1) & (n == 2))).astype(np.int64)
        frames.append(g[size // 2 - 1: size // 2 + 1].reshape(-1))
    a = np.stack(frames)
    return a[:, ::2] * 2 + a[:, 1::2]           # 4 labels from pairs of cells


def lorenz(n: int = 20000, dt: float = 0.01, sigma=10.0, rho=28.0, beta=8 / 3, x0=(1., 1., 1.)) -> np.ndarray:
    x = np.array(x0, dtype=np.float64)
    out = np.empty((n, 3))
    for i in range(n):
        dx = np.array([sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]])
        x = x + dt * dx
        out[i] = x
    return (out - out.mean(0)) / out.std(0)


def levy_flight(n: int = 20000, alpha: float = 1.5, dim: int = 2, seed: int = 0) -> np.ndarray:
    """Unit direction + Pareto step length (power-law), as a (n, dim+1) sequence."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, dim))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    step = rng.pareto(alpha, size=(n, 1)) + 1.0
    return np.concatenate([d, np.log(step)], axis=1)


def quantize_image(img: np.ndarray, n_colors: int = 4, seed: int = 0) -> np.ndarray:
    """Colour-quantise an RGB image (H, W, 3) to n_colors labels with a few k-means steps."""
    px = img.reshape(-1, 3).astype(np.float64)
    rng = np.random.default_rng(seed)
    c = px[rng.choice(len(px), n_colors, replace=False)]
    for _ in range(15):
        lab = np.argmin(((px[:, None] - c[None]) ** 2).sum(-1), axis=1)
        c = np.stack([px[lab == k].mean(0) if np.any(lab == k) else c[k] for k in range(n_colors)])
    return lab.reshape(img.shape[:2])


# ---------------------------------------------------------------- controls
def shuffled(a: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same label/value histogram, structure destroyed (Free Lunch control)."""
    flat = a.reshape(-1, *a.shape[2:]) if a.ndim > 2 else a.reshape(-1)
    flat = flat.copy()
    np.random.default_rng(seed).shuffle(flat)
    return flat.reshape(a.shape)


# ---------------------------------------------------------------- streams
class PatternStream:
    """Sequential reader over a pattern, emitting fixed-size vectors."""

    def __init__(self, kind: str, data: np.ndarray, window: int = 4, n_labels: int | None = None):
        self.kind = kind
        if data.ndim == 2 and np.issubdtype(data.dtype, np.integer):     # label image
            self.labels = data.reshape(-1)
            self.n_labels = int(n_labels or (self.labels.max() + 1))
            self.seq = None
            self.dim = window * self.n_labels
        else:                                                            # continuous sequence
            self.seq = data.reshape(len(data), -1).astype(np.float32)
            self.labels = None
            self.dim = window * self.seq.shape[1]
        self.window = window
        self.length = len(self.labels) if self.labels is not None else len(self.seq)
        self.pos = 0

    def reset(self, offset: int = 0) -> None:
        self.pos = offset % self.length

    def peek(self, offset: int = 0) -> np.ndarray:
        idx = (self.pos + offset + np.arange(self.window)) % self.length
        if self.labels is not None:
            v = np.zeros((self.window, self.n_labels), dtype=np.float32)
            v[np.arange(self.window), self.labels[idx]] = 1.0
            return v.reshape(-1)
        return self.seq[idx].reshape(-1)

    def next(self) -> np.ndarray:
        v = self.peek()
        self.pos = (self.pos + 1) % self.length
        return v


def make_stream(name: str, seed: int = 0, window: int = 4, art_image: np.ndarray | None = None) -> PatternStream:
    """Factory. Names: fractal, rule30, life, lorenz, levy, art, noise_labels, noise_gauss,
    and any of those with suffix '_shuffled'."""
    base = name.removesuffix('_shuffled')
    if base == 'fractal':
        rng = np.random.default_rng(seed)
        data = halley_fractal(a=int(rng.integers(2, 7)), b=int(rng.integers(1, 13)))
    elif base == 'rule30':
        data = rule30(seed=seed) * 2 + np.roll(rule30(seed=seed), 1, axis=1)
    elif base == 'life':
        data = game_of_life(seed=seed)
    elif base == 'lorenz':
        data = lorenz()
    elif base == 'levy':
        data = levy_flight(seed=seed)
    elif base == 'art':
        assert art_image is not None, 'art needs an RGB image'
        data = quantize_image(art_image, seed=seed)
    elif base == 'noise_labels':
        data = np.random.default_rng(seed).integers(0, 4, size=(128, 128))
    elif base == 'noise_gauss':
        data = np.random.default_rng(seed).normal(size=(20000, 3)).astype(np.float32)
    else:
        raise ValueError(name)
    if name.endswith('_shuffled'):
        data = shuffled(data, seed=seed + 1)
    return PatternStream(name, data, window=window)
