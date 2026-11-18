"""Continuous embedding of Craftax-Classic's 17 discrete actions (so gradient planners can optimise them).

vector layout (14-d):  [dx, dy | do, sleep | place_stone, place_table, place_furnace, place_plant |
                        make_wood_pickaxe, make_stone_pickaxe, make_iron_pickaxe,
                        make_wood_sword, make_stone_sword, make_iron_sword]
Movement is geometric (LEFT=(-1,0), RIGHT=(1,0), UP=(0,-1), DOWN=(0,1)); NOOP is the zero vector.
"""
import numpy as np

NAMES = ['noop', 'left', 'right', 'up', 'down', 'do', 'sleep', 'place_stone', 'place_table', 'place_furnace',
         'place_plant', 'make_wood_pickaxe', 'make_stone_pickaxe', 'make_iron_pickaxe', 'make_wood_sword',
         'make_stone_sword', 'make_iron_sword']
DIM = 14
PROTOS = np.zeros((17, DIM), dtype=np.float32)
PROTOS[1, 0] = -1; PROTOS[2, 0] = 1; PROTOS[3, 1] = -1; PROTOS[4, 1] = 1
for i, a in enumerate(range(5, 17)):
    PROTOS[a, 2 + i] = 1.0


def embed(actions: np.ndarray) -> np.ndarray:
    return PROTOS[np.asarray(actions, dtype=np.int64)]


def decode(vecs: np.ndarray) -> np.ndarray:
    """Nearest prototype (Euclidean). vecs (..., 14) -> int actions (...)."""
    v = np.asarray(vecs, dtype=np.float32)
    d = ((v[..., None, :] - PROTOS) ** 2).sum(-1)
    return d.argmin(-1)
