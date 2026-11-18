"""Grid maze from Ertle, Levin & Scheutz (2025), "Free Lunch? Low-Cost Intelligence Through
Pattern-Guided Exploration".

- Kruskal maze on an n x n cell grid (n odd), scaled to path width `path_scale` (their Alg. 1).
- 4 actions N, E, S, W. For world models / gradient planners, actions are also exposed as
  continuous 2D vectors (ACTION_VECS); `continuous_to_discrete` snaps a vector to the nearest action.
- Metric: percentage of open cells visited (their exploration metric).
- Observation: top-down RGB image (white path, black wall, blue agent, green visited),
  plus the agent's own 4-neighbour sensor reading (their environmental sensors).
"""
from __future__ import annotations

import numpy as np

ACTIONS = ['N', 'E', 'S', 'W']
ACTION_VECS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)   # (dx, dy), y down


def continuous_to_discrete(v: np.ndarray) -> int:
    v = np.asarray(v, dtype=np.float32).reshape(-1)[:2]
    return int(np.argmax(ACTION_VECS @ v))


def kruskal_maze(n: int = 11, seed: int = 0) -> np.ndarray:
    """n x n array (n odd), 0 = path, 1 = wall."""
    assert n % 2 == 1
    rng = np.random.default_rng(seed)
    m = np.ones((n, n), dtype=np.int8)
    cells = [(i, j) for i in range(1, n, 2) for j in range(1, n, 2)]
    parent = {c: c for c in cells}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    edges = [((i, j), (i, j + 2)) for i, j in cells if j + 2 < n] + \
            [((i, j), (i + 2, j)) for i, j in cells if i + 2 < n]
    rng.shuffle(edges)
    for c in cells:
        m[c] = 0
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            m[(u[0] + v[0]) // 2, (u[1] + v[1]) // 2] = 0
    return m


def scale_paths(maze: np.ndarray, path_scale: int = 2) -> np.ndarray:
    """Widen corridors: each maze unit becomes path_scale cells, walls stay 1 thick (their Alg. 1)."""
    if path_scale == 1:
        return maze
    n = maze.shape[0]
    sizes = [1 if i % 2 == 0 else path_scale for i in range(n)]
    rows = np.repeat(np.arange(n), sizes)
    return maze[np.ix_(rows, rows)]


class FreeLunchMaze:
    """Minimal gym-like interface: reset() -> obs, step(a) -> obs, reward, done, info."""

    def __init__(self, n: int = 11, path_scale: int = 2, seed: int = 0, max_steps: int = 300,
                 render_size: int = 64, sensor_range: int = 1):
        self.n, self.path_scale, self.max_steps = n, path_scale, max_steps
        self.render_size, self.sensor_range = render_size, sensor_range
        self.set_maze(seed)

    def set_maze(self, seed: int) -> None:
        self.seed = seed
        self.grid = scale_paths(kruskal_maze(self.n, seed), self.path_scale)
        self.open = np.argwhere(self.grid == 0)

    def reset(self, start: tuple[int, int] | None = None):
        self.pos = np.array(start if start is not None else self.open[0])
        self.visited = np.zeros_like(self.grid, dtype=bool)
        self.visited[tuple(self.pos)] = True
        self.t = 0
        return self.obs()

    def step(self, a):
        if not np.isscalar(a) and np.asarray(a).size > 1:
            a = continuous_to_discrete(a)
        dx, dy = ACTION_VECS[int(a)].astype(int)
        nxt = self.pos + np.array([dy, dx])
        moved = self.grid[tuple(nxt)] == 0
        if moved:
            self.pos = nxt
        new = not self.visited[tuple(self.pos)]
        self.visited[tuple(self.pos)] = True
        self.t += 1
        done = self.t >= self.max_steps
        return self.obs(), float(new), done, {'moved': bool(moved), 'coverage': self.coverage(),
                                              'pos': self.pos.copy()}

    def coverage(self) -> float:
        return float(self.visited[self.grid == 0].mean())

    def sensors(self) -> np.ndarray:
        """Wall/path/visited reading in each cardinal direction up to sensor_range (their 4 sensors)."""
        out = []
        for dx, dy in ACTION_VECS.astype(int):
            for k in range(1, self.sensor_range + 1):
                p = self.pos + k * np.array([dy, dx])
                inside = (0 <= p[0] < self.grid.shape[0]) and (0 <= p[1] < self.grid.shape[1])
                if not inside or self.grid[tuple(p)] == 1:
                    out.append(0)                       # wall
                elif self.visited[tuple(p)]:
                    out.append(2)                       # visited path
                else:
                    out.append(1)                       # unvisited path
        return np.array(out, dtype=np.int64)

    def render(self) -> np.ndarray:
        img = np.where(self.grid[..., None] == 1, 0, 255).astype(np.uint8).repeat(3, -1)
        img[self.visited & (self.grid == 0)] = (0, 200, 0)
        img[tuple(self.pos)] = (0, 0, 255)
        s = self.render_size // self.grid.shape[0] or 1
        img = img.repeat(s, 0).repeat(s, 1)
        pad = self.render_size - img.shape[0]
        if pad > 0:
            img = np.pad(img, ((0, pad), (0, pad), (0, 0)))
        return img[:self.render_size, :self.render_size]

    def obs(self) -> dict:
        return {'pixels': self.render(), 'sensors': self.sensors(), 'pos': self.pos.copy()}
