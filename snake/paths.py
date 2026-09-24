"""Time-aware path finding, shared by the referee (oracle.py) and the brain's senses.

"Time-aware": the snake's body is not a fixed wall. A body cell with stamp s frees up
once the tail has passed it, which is  s + length - t  moves from now. A cell only
blocks the head if it will still be body at the moment the head would arrive.

Escaping a box: a snake shut in by its own body is not necessarily doomed. If the
space it is in is big enough to keep moving around until the nearest wall of body
moves away, it can wait it out. So a position counts as escapable when

    moves until the nearest blocking body cell opens  <=  arrival time + room to move in
"""

import numpy as np
from numba import njit

from .game import DELTAS

DY = np.ascontiguousarray(DELTAS[:, 0])
DX = np.ascontiguousarray(DELTAS[:, 1])
UNREACHABLE = np.int64(1 << 40)


@njit(cache=True)
def bfs(stamp, t, length, g, sy, sx, d0, ty, tx, dy, dx, visited, queue, qdist, parent):
    """Time-aware BFS from (sy, sx), where the head arrives at time d0.

    Returns (arrival time at (ty, tx) or UNREACHABLE, cells reached, soonest time a
    blocking body cell next to the reached area opens, or UNREACHABLE if none).
    parent[] records the path as flat indices. ty = -1 floods the whole area.
    """
    if sy == ty and sx == tx:
        return d0, 1, UNREACHABLE
    visited[:, :] = False
    visited[sy, sx] = True
    parent[sy * g + sx] = -1
    queue[0, 0] = sy
    queue[0, 1] = sx
    qdist[0] = d0
    head, tail = 0, 1
    opens = UNREACHABLE
    while head < tail:
        y = queue[head, 0]
        x = queue[head, 1]
        d = qdist[head]
        head += 1
        for k in range(4):
            yy = y + dy[k]
            xx = x + dx[k]
            if yy < 0 or yy >= g or xx < 0 or xx >= g or visited[yy, xx]:
                continue
            free_at = stamp[yy, xx] + length - t
            if free_at > d + 1:
                if free_at < opens:              # body in the way for now; note when it opens
                    opens = free_at
                continue
            parent[yy * g + xx] = y * g + x
            if yy == ty and xx == tx:
                return d + 1, tail + 1, opens
            visited[yy, xx] = True
            queue[tail, 0] = yy
            queue[tail, 1] = xx
            qdist[tail] = d + 1
            tail += 1
    return UNREACHABLE, tail, opens


@njit(cache=True)
def escape(stamp, t, length, g, sy, sx, d0, dy, dx, visited, queue, qdist, parent):
    """Can a head arriving at (sy, sx) at time d0 keep moving forever?

    Returns (escapable, room, moves until the nearest wall of body opens (0 = none)).
    """
    _, room, opens = bfs(stamp, t, length, g, sy, sx, d0, -1, -1, dy, dx, visited, queue, qdist, parent)
    if opens == UNREACHABLE:
        return True, room, 0
    wait = max(0, opens - d0)
    return opens <= d0 + room, room, wait


@njit(cache=True)
def safe_after_eating(stamp, t, length, g, sy, sx, ay, ax, dist, parent, vstamp,
                      dy, dx, visited, queue, qdist, parent2):
    """The head went (sy, sx) -> ... -> apple along parent[], arriving at time dist.
    Replay that on a copy of the board, grow by one, and check the snake can escape."""
    vstamp[:, :] = stamp
    cell = ay * g + ax
    j = dist
    while True:                                  # walk the path backwards from the apple
        vstamp[cell // g, cell % g] = t + j
        if cell == sy * g + sx:
            break
        cell = parent[cell]
        j -= 1
    ok, _, _ = escape(vstamp, t + dist, length + 1, g, ay, ax, 0, dy, dx, visited, queue, qdist, parent2)
    return ok
