"""A fixed route through every square of the board (a Hamiltonian cycle).

This is how Snake is actually beaten: if the snake only ever moves along one closed
route that visits every square once, its body always lies along that route in order,
so it can never trap itself, and it fills the board. Following the route blindly is
slow, so good players take shortcuts towards the apple, but only ones that keep the
body in route order with enough room left before the tail.

The route (even board sizes only; an odd x odd board has no such route):

    row 0 left to right, then a serpentine up and down columns g-1 .. 1 (rows 1 .. g-1),
    then back up column 0 to the start.

It is flipped if needed so the snake's starting body already lies along it in order.
"""

from functools import lru_cache

import numpy as np

from . import config as C


@lru_cache(maxsize=None)
def route(g):
    """(g, g) array: each square's position along the route, or all -1 if g is odd."""
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
    # Every snake starts at the centre facing up with its body below the head: make the
    # route run tail -> head there, so the starting body is already in route order.
    n = g * g
    head, neck = order[g // 2, g // 2], order[g // 2 + 1, g // 2]
    if (head - neck) % n != 1:
        order = (-order) % n
    order.setflags(write=False)
    return order


def route_cells(g):
    """(g*g, 2): the squares in route order."""
    order = route(g)
    cells = np.zeros((g * g, 2), dtype=np.int64)
    ys, xs = np.nonzero(order >= 0)
    cells[order[ys, xs]] = np.stack([ys, xs], 1)
    return cells


def buffer():
    return C.CYCLE_BUFFER
