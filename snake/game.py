import numpy as np
from numba import njit

from . import config as C

DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int64)

TURN_LEFT, STRAIGHT, TURN_RIGHT = 0, 1, 2
ACTION_TURN = np.array([-1, 0, 1], dtype=np.int64)

EMPTY = np.int64(-(10**12))

ALIVE, DIED_WALL, DIED_SELF, DIED_STARVED, DIED_WON = 0, 1, 2, 3, 4
FINISHED = 5
REL_EMPTY = -(2**30)


@njit(cache=True)
def _mix(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(cache=True)
def _place_apples(stamp, t, length, seeds, placed, apple, idx, g):
    ok = np.ones(len(idx), dtype=np.bool_)
    for m in range(len(idx)):
        i = idx[m]
        free_after = t[i] - length[i]
        base = _mix(_mix(np.uint64(seeds[i])) ^ np.uint64(placed[i]))
        placed[i] += 1
        found = False
        for j in range(64):
            h = _mix(base ^ np.uint64(j))
            y = np.int64(h % np.uint64(g))
            x = np.int64((h >> np.uint64(32)) % np.uint64(g))
            if stamp[i, y, x] <= free_after:
                found = True
                break
        if not found:
            count = 0
            for yy in range(g):
                for xx in range(g):
                    if stamp[i, yy, xx] <= free_after:
                        count += 1
            if count == 0:
                ok[m] = False
                continue
            pick = np.int64(_mix(base ^ np.uint64(0xFFFFFFFF)) % np.uint64(count))
            for yy in range(g):
                for xx in range(g):
                    if stamp[i, yy, xx] <= free_after:
                        if pick == 0:
                            y, x = yy, xx
                        pick -= 1
        apple[i, 0] = y
        apple[i, 1] = x
    return ok


class Snakes:
    def __init__(self, n, grid, master_rng=None):
        if grid < 8:
            raise ValueError("grid must be at least 8")
        self.n = n
        self.size = grid
        self.master_rng = master_rng if master_rng is not None else np.random.default_rng()

        self.stamp = np.full((n, grid, grid), EMPTY, dtype=np.int64)
        self.t = np.zeros(n, dtype=np.int64)
        self.length = np.zeros(n, dtype=np.int64)
        self.head = np.zeros((n, 2), dtype=np.int64)
        self.dir = np.zeros(n, dtype=np.int64)
        self.apple = np.zeros((n, 2), dtype=np.int64)
        self.alive = np.zeros(n, dtype=bool)
        self.score = np.zeros(n, dtype=np.int64)
        self.hunger = np.zeros(n, dtype=np.int64)
        self.budget = np.zeros(n, dtype=np.int64)
        self.death = np.zeros(n, dtype=np.int64)
        self.apples_placed = np.zeros(n, dtype=np.int64)
        self.seeds = np.zeros(n, dtype=np.int64)


    def reset(self, seeds=None):
        if seeds is None:
            seeds = self.master_rng.integers(0, 2**62, size=self.n)
        self.seeds = np.asarray(seeds, dtype=np.int64).copy()

        self.stamp.fill(EMPTY)
        self.t.fill(0)
        self.length.fill(C.START_LENGTH)
        self.score.fill(0)
        self.death.fill(ALIVE)
        self.alive.fill(True)
        self.apples_placed.fill(0)

        y, x = self.size // 2, self.size // 2
        self.dir.fill(0)
        self.head[:] = (y, x)
        for k in range(C.START_LENGTH):
            self.stamp[:, y + k, x] = -k
        self._place_apples(np.arange(self.n))


    def snapshot(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        rel = self.stamp[idx] - self.t[idx][:, None, None]
        rel[rel < -(self.size * self.size + 1)] = REL_EMPTY
        return dict(stamp=rel.astype(np.int32), length=self.length[idx].copy(),
                    head=self.head[idx].copy(), dir=self.dir[idx].copy(), score=self.score[idx].copy())

    def restore(self, idx, saved, picks, seeds):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return
        stamp = saved["stamp"][picks].astype(np.int64)
        stamp[stamp == REL_EMPTY] = EMPTY
        self.stamp[idx] = stamp
        self.t[idx] = 0
        self.length[idx] = saved["length"][picks]
        self.head[idx] = saved["head"][picks]
        self.dir[idx] = saved["dir"][picks]
        self.score[idx] = saved["score"][picks]
        self.alive[idx] = True
        self.death[idx] = ALIVE
        self.seeds[idx] = seeds
        self.apples_placed[idx] = 0
        for i in self._place_apples(idx):
            self._kill(int(i), DIED_WON)

    def finish(self, idx):
        self.alive[idx] = False
        self.death[idx] = FINISHED

    @classmethod
    def copies(cls, src, idx, reps, seeds):
        new = cls.__new__(cls)
        rows = np.repeat(np.asarray(idx, dtype=np.int64), reps)
        new.n, new.size, new.master_rng = len(rows), src.size, src.master_rng
        for name in ("stamp", "t", "length", "head", "dir", "apple", "alive", "score", "hunger",
                     "budget", "death", "apples_placed"):
            setattr(new, name, getattr(src, name)[rows])
        new.seeds = np.asarray(seeds, dtype=np.int64)
        return new

    def take(self, i, src, j):
        for name in ("stamp", "t", "length", "head", "dir", "apple", "alive", "score", "hunger",
                     "budget", "death", "apples_placed"):
            getattr(self, name)[i] = getattr(src, name)[j]

    def set_snake(self, i, body, direction, apple):
        self.stamp[i].fill(EMPTY)
        self.t[i] = 0
        self.length[i] = len(body)
        for k, (y, x) in enumerate(body):
            self.stamp[i, y, x] = -k
        self.head[i] = body[0]
        self.dir[i] = direction
        self.apple[i] = apple
        self.alive[i] = True
        self.death[i] = ALIVE
        self.score[i] = 0
        self.hunger[i] = 0
        self.budget[i] = self._budget_for(i)
        self.apples_placed[i] = 1


    def body_mask(self, i):
        return self.stamp[i] > self.t[i] - self.length[i]

    def occupied_next(self, i, cells):
        ys, xs = cells[:, 0], cells[:, 1]
        return self.stamp[i, ys, xs] > self.t[i] + 1 - self.length[i]

    def safe_moves(self):
        g = self.size
        rows = np.arange(self.n)
        safe = np.zeros((self.n, 3), dtype=bool)
        for a in range(3):
            d = (self.dir + ACTION_TURN[a]) % 4
            y = self.head[:, 0] + DELTAS[d, 0]
            x = self.head[:, 1] + DELTAS[d, 1]
            inside = (y >= 0) & (y < g) & (x >= 0) & (x < g)
            cy, cx = np.clip(y, 0, g - 1), np.clip(x, 0, g - 1)
            eats = (cy == self.apple[:, 0]) & (cx == self.apple[:, 1])
            body = self.stamp[rows, cy, cx] > self.t + 1 - (self.length + eats)
            safe[:, a] = inside & ~body
        return safe


    def step(self, actions):
        n, g = self.n, self.size
        live = self.alive.copy()
        reward = np.zeros(n, dtype=np.float32)
        done = np.zeros(n, dtype=bool)
        ate = np.zeros(n, dtype=bool)
        if not live.any():
            return reward, done, ate

        idx = np.flatnonzero(live)
        new_dir = (self.dir[idx] + ACTION_TURN[actions[idx]]) % 4
        new_head = self.head[idx] + DELTAS[new_dir]
        ny, nx = new_head[:, 0], new_head[:, 1]

        wall = (ny < 0) | (ny >= g) | (nx < 0) | (nx >= g)
        cy, cx = np.clip(ny, 0, g - 1), np.clip(nx, 0, g - 1)
        eat = ~wall & (ny == self.apple[idx, 0]) & (nx == self.apple[idx, 1])
        new_len = self.length[idx] + eat
        new_t = self.t[idx] + 1
        body = ~wall & (self.stamp[idx, cy, cx] > new_t - new_len)
        crash = wall | body

        old_dist = np.abs(self.head[idx] - self.apple[idx]).sum(1)
        new_dist = np.abs(new_head - self.apple[idx]).sum(1)

        ok = ~crash
        mi = idx[ok]
        self.stamp[mi, ny[ok], nx[ok]] = new_t[ok]
        self.head[mi] = new_head[ok]
        self.dir[mi] = new_dir[ok]
        self.t[mi] = new_t[ok]
        self.length[mi] = new_len[ok]
        self.hunger[mi] += 1

        r = np.where(new_dist < old_dist, C.REWARD_CLOSER, C.REWARD_FURTHER).astype(np.float32)
        if C.SHAPING_FADES:
            r *= np.maximum(0.0, 1.0 - new_len / (g * g)).astype(np.float32)

        eaters = idx[ok & eat]
        r[eat & ok] = C.REWARD_APPLE
        ate[eaters] = True
        self.score[eaters] += 1
        won = self._place_apples(eaters)
        for i in won:
            self._kill(int(i), DIED_WON)
            done[i] = True

        starved = ok & ~eat & (self.hunger[idx] > self.budget[idx])
        r[starved] = C.REWARD_STARVE
        r[crash] = C.REWARD_DEATH

        reward[idx] = r
        reward[won] = C.REWARD_WIN
        for i in idx[wall]:
            self._kill(int(i), DIED_WALL)
        for i in idx[body]:
            self._kill(int(i), DIED_SELF)
        for i in idx[starved]:
            self._kill(int(i), DIED_STARVED)
        done[idx[crash | starved]] = True
        return reward, done, ate


    def _kill(self, i, cause):
        self.alive[i] = False
        self.death[i] = cause

    def _budget_for(self, i):
        dist = int(np.abs(self.head[i] - self.apple[i]).sum())
        return max(dist + C.STARVE_SLACK + C.STARVE_SLACK_PER_SEGMENT * int(self.length[i]),
                   int(C.STARVE_AREA_FACTOR * self.size * self.size))

    def _place_apples(self, idx):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return idx
        ok = _place_apples(self.stamp, self.t, self.length, self.seeds, self.apples_placed,
                           self.apple, idx, self.size)
        self.hunger[idx] = 0
        dist = np.abs(self.head[idx] - self.apple[idx]).sum(1)
        self.budget[idx] = np.maximum(
            dist + C.STARVE_SLACK + C.STARVE_SLACK_PER_SEGMENT * self.length[idx],
            int(C.STARVE_AREA_FACTOR * self.size * self.size))
        return idx[~ok]
