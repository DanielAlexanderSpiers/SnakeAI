"""Works out the best possible move for every snake and grades the move it actually made.

This is a referee, not a teacher. Its output only decides the snake's colour and
the accuracy stat. It is never fed to the brain or used in rewards.

"Best" follows the safe-greedy Snake strategy:

1. Eat by the shortest path, but only if it is SAFE: imagine the snake following
   that path and eating the apple, then check it can still escape afterwards.
2. If no move has a safe path to the apple, the best move is any move the snake can
   escape from, i.e. stay alive and wait for the body to open a way.
3. If nothing is escapable the snake is already doomed: an earlier move boxed it in.
   Moves then score at most TRAPPED (more room = a bit better), so it shows red.

"Escape" is time-aware (see paths.py): being shut in by your own body is fine if the
space you're in is big enough to keep moving until that body moves out of the way.
So an apple that spawned inside a loop of the snake is not a trap if the loop is
roomy enough to eat it and wait.

Grades per move (see config.py for the numbers):
    1.0    best move (there can be two or three equally best moves)
    0.5    safe path to the apple but 2 moves slower (0.33 for 4 slower, ...)
    STALL  stays alive, but a safe path to the apple was available
    GREEDY rushes the apple by a path it can't escape from afterwards
    TRAPPED can't escape at all: doomed
    0.0    dies immediately
"""

import numpy as np
from numba import njit, prange

from . import config as C
from .paths import DX, DY, UNREACHABLE, bfs, escape, safe_after_eating


@njit(parallel=True, cache=True)
def _grade_all(stamp, t, length, head, direction, apple, which, g, dy, dx,
               g_stall, g_greedy, g_trapped):
    n = stamp.shape[0]
    grades = np.zeros((n, 3), dtype=np.float32)
    for i in prange(n):
        if not which[i]:
            continue
        visited = np.zeros((g, g), dtype=np.bool_)
        queue = np.empty((g * g, 2), dtype=np.int64)
        qdist = np.empty(g * g, dtype=np.int64)
        parent = np.empty(g * g, dtype=np.int64)
        parent2 = np.empty(g * g, dtype=np.int64)
        vstamp = np.empty((g, g), dtype=np.int64)

        L = length[i]
        ay, ax = apple[i, 0], apple[i, 1]
        safe = np.zeros(3, dtype=np.bool_)
        dist = np.full(3, UNREACHABLE, dtype=np.int64)
        apple_safe = np.zeros(3, dtype=np.bool_)
        survive = np.zeros(3, dtype=np.bool_)
        room = np.zeros(3, dtype=np.int64)
        for a in range(3):
            d = (direction[i] + a - 1) % 4
            y = head[i, 0] + dy[d]
            x = head[i, 1] + dx[d]
            if y < 0 or y >= g or x < 0 or x >= g:
                continue
            grow = 1 if (y == ay and x == ax) else 0
            if stamp[i, y, x] > t[i] + 1 - (L + grow):
                continue
            safe[a] = True
            dist[a], _, _ = bfs(stamp[i], t[i], L, g, y, x, 1, ay, ax, dy, dx,
                                visited, queue, qdist, parent)
            if dist[a] < UNREACHABLE:
                apple_safe[a] = safe_after_eating(stamp[i], t[i], L, g, y, x, ay, ax, dist[a],
                                                  parent, vstamp, dy, dx, visited, queue, qdist, parent2)
            ok, room[a], _ = escape(stamp[i], t[i], L, g, y, x, 1, dy, dx, visited, queue, qdist, parent2)
            survive[a] = apple_safe[a] or (grow == 0 and ok)

        best = UNREACHABLE
        for a in range(3):
            if apple_safe[a] and dist[a] < best:
                best = dist[a]
        any_survive = survive[0] or survive[1] or survive[2]

        if best < UNREACHABLE or any_survive:
            for a in range(3):
                if not safe[a]:
                    continue
                if apple_safe[a]:
                    grades[i, a] = 1.0 / (1.0 + (dist[a] - best) / 2.0)
                elif survive[a]:
                    grades[i, a] = g_stall if best < UNREACHABLE else 1.0
                elif dist[a] < UNREACHABLE:
                    grades[i, a] = g_greedy
                else:
                    grades[i, a] = g_trapped
        else:
            most = room.max()
            for a in range(3):
                if safe[a] and most > 0:
                    grades[i, a] = g_trapped * room[a] / most
    return grades


def grade_moves(snakes, which=None):
    """Grades for all three possible moves: float32 array (n, 3).

    which: bool mask of snakes to grade (default: every living snake). Others get zeros.
    """
    if which is None:
        which = snakes.alive
    return _grade_all(snakes.stamp, snakes.t, snakes.length, snakes.head, snakes.dir,
                      snakes.apple, which, snakes.size, DY, DX,
                      np.float32(C.GRADE_STALL), np.float32(C.GRADE_GREEDY),
                      np.float32(C.GRADE_TRAPPED))
