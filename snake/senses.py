import numpy as np
from numba import njit, prange

from . import config as C
from .cycle import route
from .game import DELTAS
from .paths import UNREACHABLE, safe_after_eating

ROUTE_ORDER = 80
SENSES = {"basic": 11, "vision": 42, "vision2": 51, "vision3": 57, "vision4": 66, "vision5": 81}
DEFAULT_SENSES = "vision5"
LEVEL = {"vision": 1, "vision2": 2, "vision3": 3, "vision4": 4, "vision5": 5}


def n_inputs(kind):
    return SENSES[kind]


def observe(snakes, kind="basic"):
    if kind == "basic":
        return _observe_basic(snakes)
    if kind in LEVEL:
        out = np.zeros((snakes.n, SENSES[kind]), dtype=np.float32)
        _vision(snakes.stamp, snakes.t, snakes.length, snakes.head, snakes.dir, snakes.apple,
                snakes.alive, snakes.size, _DY, _DX, LEVEL[kind], route(snakes.size),
                C.CYCLE_BUFFER, out)
        return out
    raise ValueError(f"unknown senses '{kind}'")


def fenced_off(snakes):
    out = np.zeros(snakes.n, dtype=np.float32)
    _fenced_all(snakes.stamp, snakes.t, snakes.length, snakes.head, snakes.alive, snakes.size,
                _DY, _DX, out)
    return out


def _observe_basic(snakes):
    n, g = snakes.n, snakes.size
    rows = np.arange(n)
    out = np.zeros((n, 11), dtype=np.float32)

    for col, turn in enumerate((0, -1, 1)):
        d = (snakes.dir + turn) % 4
        cell = snakes.head + DELTAS[d]
        y, x = cell[:, 0], cell[:, 1]
        wall = (y < 0) | (y >= g) | (x < 0) | (x >= g)
        cy, cx = np.clip(y, 0, g - 1), np.clip(x, 0, g - 1)
        body = snakes.stamp[rows, cy, cx] > snakes.t + 1 - snakes.length
        out[:, col] = wall | body

    out[rows, 3 + snakes.dir] = 1.0

    hy, hx = snakes.head[:, 0], snakes.head[:, 1]
    ay, ax = snakes.apple[:, 0], snakes.apple[:, 1]
    out[:, 7] = ay < hy
    out[:, 8] = ay > hy
    out[:, 9] = ax < hx
    out[:, 10] = ax > hx
    return out


_DY = np.ascontiguousarray(DELTAS[:, 0])
_DX = np.ascontiguousarray(DELTAS[:, 1])
_RAYS = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], dtype=np.int64)


@njit(cache=True)
def _survey(stamp, t, length, g, sy, sx, ay, ax, ty, tx, dy, dx, visited, queue, qdist, parent):
    visited[:, :] = False
    visited[sy, sx] = True
    parent[sy * g + sx] = -1
    queue[0, 0] = sy
    queue[0, 1] = sx
    qdist[0] = 1
    head, tail = 0, 1
    apple_d = 1 if (sy == ay and sx == ax) else -1
    tail_d = -1
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
                if free_at < opens:
                    opens = free_at
                continue
            parent[yy * g + xx] = y * g + x
            if yy == ty and xx == tx and tail_d < 0:
                tail_d = d + 1
            if yy == ay and xx == ax and apple_d < 0:
                apple_d = d + 1
            visited[yy, xx] = True
            queue[tail, 0] = yy
            queue[tail, 1] = xx
            qdist[tail] = d + 1
            tail += 1
    return tail, apple_d, tail_d, opens


@njit(cache=True)
def _fenced(stamp, blocked_above, g, sy, sx, dy, dx, visited, queue):
    visited[:, :] = False
    visited[sy, sx] = True
    queue[0, 0] = sy
    queue[0, 1] = sx
    head, tail = 0, 1
    while head < tail:
        y = queue[head, 0]
        x = queue[head, 1]
        head += 1
        for k in range(4):
            yy = y + dy[k]
            xx = x + dx[k]
            if yy < 0 or yy >= g or xx < 0 or xx >= g or visited[yy, xx]:
                continue
            if stamp[yy, xx] > blocked_above:
                continue
            visited[yy, xx] = True
            queue[tail, 0] = yy
            queue[tail, 1] = xx
            tail += 1
    return tail - 1


@njit(cache=True)
def _count_pieces(stamp, blocked_above, g, visited, queue, dy, dx):
    pieces = 0
    for sy in range(g):
        for sx in range(g):
            if visited[sy, sx] or stamp[sy, sx] > blocked_above:
                continue
            pieces += 1
            visited[sy, sx] = True
            queue[0, 0] = sy
            queue[0, 1] = sx
            head, tail = 0, 1
            while head < tail:
                y = queue[head, 0]
                x = queue[head, 1]
                head += 1
                for k in range(4):
                    yy = y + dy[k]
                    xx = x + dx[k]
                    if yy < 0 or yy >= g or xx < 0 or xx >= g or visited[yy, xx]:
                        continue
                    if stamp[yy, xx] > blocked_above:
                        continue
                    visited[yy, xx] = True
                    queue[tail, 0] = yy
                    queue[tail, 1] = xx
                    tail += 1
    return pieces


@njit(cache=True)
def _route_senses(stamp, T, L, hy, hx, ay, ax, facing, g, dy, dx, cyc, cbuf, row):
    n = g * g
    pos = np.empty(L, dtype=np.int64)
    base = T - L
    for yy in range(g):
        for xx in range(g):
            s = stamp[yy, xx]
            if s > base:
                pos[s - base - 1] = cyc[yy, xx]
    tail_at, head_at = pos[0], pos[L - 1]
    in_order = True
    prev = 0
    for k in range(1, L):
        dk = (pos[k] - tail_at) % n
        if dk <= prev:
            in_order = False
            break
        prev = dk
    apple_at = cyc[ay, ax]
    free_ahead = (tail_at - head_at) % n
    row[78] = free_ahead / n
    row[79] = ((apple_at - head_at) % n) / n
    row[80] = 1.0 if in_order else 0.0
    for a in range(3):
        if row[a * 5] == 1.0:
            continue
        d = (facing + a - 1) % 4
        y = hy + dy[d]
        x = hx + dx[d]
        c = 66 + a * 4
        jump = (cyc[y, x] - head_at) % n
        row[c] = jump / n
        row[c + 1] = 1.0 if jump == 1 else 0.0
        if in_order and (jump == 1 or jump < free_ahead - cbuf):
            row[c + 2] = 1.0
        row[c + 3] = ((apple_at - cyc[y, x]) % n) / n


@njit(cache=True)
def _fenced_one(stamp, t, length, head, i, g, dy, dx):
    free = g * g - length[i]
    if free <= 0:
        return 0.0
    visited = np.zeros((g, g), dtype=np.bool_)
    queue = np.empty((g * g, 2), dtype=np.int64)
    reached = _fenced(stamp[i], t[i] - length[i], g, head[i, 0], head[i, 1], dy, dx, visited, queue)
    return (free - reached) / free


@njit(parallel=True, cache=True)
def _fenced_all(stamp, t, length, head, alive, g, dy, dx, out):
    for i in prange(stamp.shape[0]):
        if alive[i]:
            out[i] = _fenced_one(stamp, t, length, head, i, g, dy, dx)


@njit(cache=True)
def _vision_one(stamp, t, length, head, direction, apple, i, g, dy, dx, level, cyc, cbuf, out):
    visited = np.zeros((g, g), dtype=np.bool_)
    queue = np.empty((g * g, 2), dtype=np.int64)
    qdist = np.empty(g * g, dtype=np.int64)
    parent = np.empty(g * g, dtype=np.int64)
    parent2 = np.empty(g * g, dtype=np.int64)
    vstamp = np.empty((g, g), dtype=np.int64)
    L = length[i]
    T = t[i]
    hy, hx = head[i, 0], head[i, 1]
    ay, ax = apple[i, 0], apple[i, 1]
    free_cells = max(1, g * g - L)

    ty, tx = -1, -1
    want = T + 2 - L
    for yy in range(g):
        for xx in range(g):
            if stamp[i, yy, xx] == want:
                ty, tx = yy, xx

    for a in range(3):
        d = (direction[i] + a - 1) % 4
        y = hy + dy[d]
        x = hx + dx[d]
        c = a * 5
        if y < 0 or y >= g or x < 0 or x >= g:
            out[i, c] = 1.0
            continue
        grow = 1 if (y == ay and x == ax) else 0
        if stamp[i, y, x] > T + 1 - (L + grow):
            out[i, c] = 1.0
            continue
        room, apple_d, tail_d, opens = _survey(stamp[i], T, L, g, y, x, ay, ax, ty, tx,
                                               dy, dx, visited, queue, qdist, parent)
        tail_ok = tail_d > 0
        out[i, c + 1] = room / free_cells
        out[i, c + 2] = min(1.0, room / (L + 1.0))
        out[i, c + 3] = 1.0 if tail_ok else 0.0
        if apple_d > 0:
            out[i, c + 4] = 1.0 - (apple_d - 1) / (2.0 * g)

        if level >= 2:
            c2 = 42 + a * 3
            eat_safe = False
            if apple_d > 0:
                eat_safe = safe_after_eating(stamp[i], T, L, g, y, x, ay, ax, apple_d, parent,
                                             vstamp, dy, dx, visited, queue, qdist, parent2)
            if grow == 1:
                ok, wait = eat_safe, 0
            elif opens == UNREACHABLE:
                ok, wait = True, 0
            else:
                ok, wait = opens <= 1 + room, max(0, opens - 1)
            out[i, c2] = 1.0 if ok else 0.0
            out[i, c2 + 1] = min(1.0, wait / (L + 1.0))
            out[i, c2 + 2] = 1.0 if eat_safe else 0.0

        if level >= 3:
            c3 = 51 + a * 2
            after = T + 1 - (L + grow)
            blocked = 0
            for k in range(4):
                yy = y + dy[k]
                xx = x + dx[k]
                if yy == hy and xx == hx:
                    continue
                if yy < 0 or yy >= g or xx < 0 or xx >= g or stamp[i, yy, xx] > after:
                    blocked += 1
            out[i, c3] = blocked / 3.0
            free_after = g * g - (L + grow)
            if free_after > 0:
                saved = stamp[i, y, x]
                stamp[i, y, x] = after + 1
                reached = _fenced(stamp[i], after, g, y, x, dy, dx, visited, queue)
                out[i, c3 + 1] = (free_after - reached) / free_after
                if level >= 4:
                    c4 = 57 + a * 3
                    if tail_ok:
                        out[i, c4] = 1.0 / (1.0 + (tail_d - 1) / 4.0)
                    if grow == 1 or visited[ay, ax]:
                        out[i, c4 + 1] = 1.0
                    pieces = _count_pieces(stamp[i], after, g, visited, queue, dy, dx)
                    out[i, c4 + 2] = min(1.0, pieces / 3.0)
                stamp[i, y, x] = saved
            elif level >= 4 and tail_ok:
                out[i, 57 + a * 3] = 1.0 / (1.0 + (tail_d - 1) / 4.0)

    f = direction[i]
    r = (f + 1) % 4
    for k in range(8):
        fy = _RAYS[k, 0] * dy[f] + _RAYS[k, 1] * dy[r]
        fx = _RAYS[k, 0] * dx[f] + _RAYS[k, 1] * dx[r]
        c = 15 + k * 3
        y, x = hy, hx
        step = 0
        moves = abs(fy) + abs(fx)
        body_seen = False
        while True:
            y += fy
            x += fx
            step += 1
            if y < 0 or y >= g or x < 0 or x >= g:
                out[i, c] = 1.0 / step
                break
            if not body_seen and stamp[i, y, x] + L - T > step * moves:
                out[i, c + 1] = 1.0 / step
                body_seen = True
            if y == ay and x == ax:
                out[i, c + 2] = 1.0 / step
    if level >= 5 and cyc[0, 0] >= 0:
        _route_senses(stamp[i], T, L, hy, hx, ay, ax, direction[i], g, dy, dx, cyc, cbuf, out[i])

    vy, vx = ay - hy, ax - hx
    out[i, 39] = (vy * dy[f] + vx * dx[f]) / g
    out[i, 40] = (vy * dy[r] + vx * dx[r]) / g
    out[i, 41] = L / (g * g)


@njit(cache=True)
def _light_one(stamp, t, length, head, direction, apple, i, g, dy, dx, cyc, cbuf, out):
    L = length[i]
    T = t[i]
    hy, hx = head[i, 0], head[i, 1]
    ay, ax = apple[i, 0], apple[i, 1]
    for a in range(3):
        d = (direction[i] + a - 1) % 4
        y = hy + dy[d]
        x = hx + dx[d]
        grow = 1 if (y == ay and x == ax) else 0
        if y < 0 or y >= g or x < 0 or x >= g or stamp[i, y, x] > T + 1 - (L + grow):
            out[i, a * 5] = 1.0
    if cyc[0, 0] >= 0 and out.shape[1] > 80:
        _route_senses(stamp[i], T, L, hy, hx, ay, ax, direction[i], g, dy, dx, cyc, cbuf, out[i])


@njit(parallel=True, cache=True)
def _vision(stamp, t, length, head, direction, apple, alive, g, dy, dx, level, cyc, cbuf, out):
    for i in prange(stamp.shape[0]):
        if alive[i]:
            _vision_one(stamp, t, length, head, direction, apple, i, g, dy, dx, level, cyc, cbuf, out)

