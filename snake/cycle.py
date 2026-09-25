from functools import lru_cache

import numpy as np

from . import config as C


@lru_cache(maxsize=None)
def route(g):
    order = np.full((g, g), -1, dtype=np.int64)
    if g % 2:
        return order
    cells = [(0, x) for x in range(g)]
    for j, col in enumerate(range(g - 1, 0, -1)):
        rows = range(1, g) if j % 2 == 0 else range(g - 1, 0, -1)
        cells += [(r, col) for r in rows]
    cells += [(r, 0) for r in range(g - 1, 0, -1)]
    for k, (y, x) in enumerate(cells):
        order[y, x] = k
    n = g * g
    head, neck = order[g // 2, g // 2], order[g // 2 + 1, g // 2]
    if (head - neck) % n != 1:
        order = (-order) % n
    order.setflags(write=False)
    return order


def route_cells(g):
    order = route(g)
    cells = np.zeros((g * g, 2), dtype=np.int64)
    ys, xs = np.nonzero(order >= 0)
    cells[order[ys, xs]] = np.stack([ys, xs], 1)
    return cells


def buffer():
    return C.CYCLE_BUFFER
