import numpy as np
from numba import njit

from . import config as C
from .game import (DELTAS, DIED_SELF, DIED_STARVED, DIED_WALL, DIED_WON, Snakes, _place_apples)
from .cycle import route
from .senses import ROUTE_ORDER, _light_one, fenced_off, n_inputs, observe

_DY = np.ascontiguousarray(DELTAS[:, 0])
_DX = np.ascontiguousarray(DELTAS[:, 1])
_STATE = ("stamp", "t", "length", "head", "dir", "apple", "alive", "score", "hunger", "budget",
          "death", "apples_placed")


@njit(cache=True)
def _step(stamp, t, length, head, direction, apple, alive, score, hunger, budget, death, placed,
          seeds, actions, g, dy, dx, r_apple, r_win, r_death, r_starve, r_closer, r_further,
          fades, slack, per_seg, area_min, reward, done):
    one = np.zeros(1, dtype=np.int64)
    for i in range(stamp.shape[0]):
        reward[i] = 0.0
        done[i] = False
        if not alive[i]:
            continue
        nd = (direction[i] + actions[i] - 1) % 4
        y = head[i, 0] + dy[nd]
        x = head[i, 1] + dx[nd]
        if y < 0 or y >= g or x < 0 or x >= g:
            alive[i] = False
            death[i] = DIED_WALL
            reward[i] = r_death
            done[i] = True
            continue
        eat = y == apple[i, 0] and x == apple[i, 1]
        new_len = length[i] + (1 if eat else 0)
        new_t = t[i] + 1
        if stamp[i, y, x] > new_t - new_len:
            alive[i] = False
            death[i] = DIED_SELF
            reward[i] = r_death
            done[i] = True
            continue
        old_dist = abs(head[i, 0] - apple[i, 0]) + abs(head[i, 1] - apple[i, 1])
        new_dist = abs(y - apple[i, 0]) + abs(x - apple[i, 1])
        stamp[i, y, x] = new_t
        head[i, 0] = y
        head[i, 1] = x
        direction[i] = nd
        t[i] = new_t
        length[i] = new_len
        hunger[i] += 1
        r = r_closer if new_dist < old_dist else r_further
        if fades:
            r = np.float32(r * np.float32(max(0.0, 1.0 - new_len / (g * g))))
        if eat:
            r = r_apple
            score[i] += 1
            one[0] = i
            ok = _place_apples(stamp, t, length, seeds, placed, apple, one, g)
            hunger[i] = 0
            dist = abs(head[i, 0] - apple[i, 0]) + abs(head[i, 1] - apple[i, 1])
            budget[i] = max(dist + slack + per_seg * length[i], area_min)
            if not ok[0]:
                alive[i] = False
                death[i] = DIED_WON
                reward[i] = r_win
                done[i] = True
                continue
        elif hunger[i] > budget[i]:
            alive[i] = False
            death[i] = DIED_STARVED
            r = r_starve
            done[i] = True
        reward[i] = r


@njit(cache=True)
def _copy_state(stamp, t, length, head, direction, apple, alive, score, hunger, budget, death, placed,
                d_stamp, d_t, d_length, d_head, d_direction, d_apple, d_alive, d_score, d_hunger,
                d_budget, d_death, d_placed, src, dst):
    for j in dst:
        d_stamp[j, :, :] = stamp[src]
        d_t[j] = t[src]
        d_length[j] = length[src]
        d_head[j, 0] = head[src, 0]
        d_head[j, 1] = head[src, 1]
        d_direction[j] = direction[src]
        d_apple[j, 0] = apple[src, 0]
        d_apple[j, 1] = apple[src, 1]
        d_alive[j] = alive[src]
        d_score[j] = score[src]
        d_hunger[j] = hunger[src]
        d_budget[j] = budget[src]
        d_death[j] = death[src]
        d_placed[j] = placed[src]


def copy_state(src_batch, src, dst_batch, dst):
    a, b = src_batch, dst_batch
    _copy_state(a.stamp, a.t, a.length, a.head, a.dir, a.apple, a.alive, a.score, a.hunger, a.budget,
                a.death, a.apples_placed, b.stamp, b.t, b.length, b.head, b.dir, b.apple, b.alive,
                b.score, b.hunger, b.budget, b.death, b.apples_placed, src, dst)


def fast_step(snakes, actions, reward, done):
    g = snakes.size
    _step(snakes.stamp, snakes.t, snakes.length, snakes.head, snakes.dir, snakes.apple, snakes.alive,
          snakes.score, snakes.hunger, snakes.budget, snakes.death, snakes.apples_placed, snakes.seeds,
          actions, g, _DY, _DX, np.float32(C.REWARD_APPLE), np.float32(C.REWARD_WIN),
          np.float32(C.REWARD_DEATH), np.float32(C.REWARD_STARVE), np.float32(C.REWARD_CLOSER),
          np.float32(C.REWARD_FURTHER), C.SHAPING_FADES, C.STARVE_SLACK, C.STARVE_SLACK_PER_SEGMENT,
          int(C.STARVE_AREA_FACTOR * g * g), reward, done)


class NumpyBrain:

    def __init__(self, agent):
        p = {k: v.detach().numpy().astype(np.float32) for k, v in agent.online.state_dict().items()}
        self.w = [np.ascontiguousarray(p[f"net.{k}.weight"].T) for k in (0, 2, 4)]
        self.b = [p[f"net.{k}.bias"] for k in (0, 2, 4)]

    def __call__(self, x):
        h = np.maximum(x @ self.w[0] + self.b[0], 0)
        h = np.maximum(h @ self.w[1] + self.b[1], 0)
        return h @ self.w[2] + self.b[2]


class FastPlayer:

    def __init__(self, agent, snake, rng):
        self.agent, self.snake, self.rng = agent, snake, rng
        self.brain = NumpyBrain(agent)
        self.sim = Snakes.copies(snake, np.zeros(1, dtype=np.int64), 3, np.zeros(3, dtype=np.int64))
        self.moves = np.arange(3, dtype=np.int64)
        self.rows = [np.array([k], dtype=np.int64) for k in range(3)]
        self.first = np.zeros(1, dtype=np.int64)
        self.r3, self.d3 = np.zeros(3, dtype=np.float32), np.zeros(3, dtype=bool)
        self.r1, self.d1 = np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool)
        self.vision5 = agent.senses == "vision5"

    def observe(self):
        s = self.snake
        if not self.agent.lookahead or not self.agent.senses.startswith("vision"):
            return observe(s, self.agent.senses)
        out = np.zeros((1, n_inputs(self.agent.senses)), dtype=np.float32)
        if s.alive[0]:
            _light_one(s.stamp, s.t, s.length, s.head, s.dir, s.apple, 0, s.size, _DY, _DX,
                       route(s.size), C.CYCLE_BUFFER, out)
        return out

    def step(self, obs):
        agent, s, sim = self.agent, self.snake, self.sim
        allowed = None
        if agent.shield:
            o = obs[0]
            allowed = [o[a * 5] != 1.0 and o[66 + a * 4 + 2] == 1.0 and o[66 + a * 4 + 3] <= o[79] + 1e-6
                       for a in range(3)]
            if allowed.count(True) == 1:
                return self._play(allowed.index(True))
            if not any(allowed):
                allowed = None
        sim_obs = None
        if agent.lookahead:
            copy_state(s, 0, sim, self.moves)
            sim.seeds[:] = self.rng.integers(0, 2**62, size=3)
            candidate = np.ones(3, dtype=bool) if allowed is None else np.array(allowed)
            sim.alive[:] = candidate
            fast_step(sim, self.moves, self.r3, self.d3)
            value = self.r3.copy()
            live = candidate & ~self.d3
            if C.COMPACT_WEIGHT:
                value -= C.COMPACT_WEIGHT * C.GAMMA * fenced_off(sim) * live
            if live.any():
                sim_obs = observe(sim, agent.senses)
                if self.vision5 and C.ROUTE_WEIGHT:
                    value += C.ROUTE_WEIGHT * C.GAMMA * sim_obs[:, ROUTE_ORDER] * live
                value[live] += C.GAMMA * self.brain(sim_obs[live]).max(1)
        else:
            value = self.brain(obs)[0]
        if allowed is not None:
            value = np.where(allowed, value, -np.inf)
        a = int(np.argmax(value))
        if agent.lookahead and sim_obs is not None and sim.score[a] == s.score[0]:
            copy_state(sim, a, s, self.first)
            return sim_obs[a:a + 1]
        return self._play(a)

    def _play(self, a):
        fast_step(self.snake, self.rows[a], self.r1, self.d1)
        return self.observe()
