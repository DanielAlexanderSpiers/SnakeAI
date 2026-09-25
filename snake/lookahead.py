import numpy as np

from . import config as C
from .game import Snakes
from .senses import ROUTE_ORDER, fenced_off, observe


def move_values(agent, snakes, idx, rng):
    return try_moves(agent, snakes, idx, rng)[0]


def try_moves(agent, snakes, idx, rng):
    m = len(idx)
    sim = Snakes.copies(snakes, idx, 3, rng.integers(0, 2**62, size=3 * m))
    reward, done, _ = sim.step(np.tile(np.arange(3), m))
    value = reward.astype(np.float32)
    live = ~done
    obs = None
    if C.COMPACT_WEIGHT:
        value -= C.COMPACT_WEIGHT * C.GAMMA * fenced_off(sim) * live
    if live.any():
        obs = observe(sim, agent.senses)
        if agent.senses == "vision5" and C.ROUTE_WEIGHT:
            value += C.ROUTE_WEIGHT * C.GAMMA * obs[:, ROUTE_ORDER] * live
        value[live] += C.GAMMA * agent.q_values(obs[live]).max(1)
    return value.reshape(m, 3), sim, obs


def route_safe(obs):
    dies = obs[:, [0, 5, 10]] == 1.0
    safe = obs[:, [66 + 2, 70 + 2, 74 + 2]] == 1.0
    closer = obs[:, [66 + 3, 70 + 3, 74 + 3]] <= obs[:, [79]] + 1e-6
    return safe & closer & ~dies


def choose_moves(agent, snakes, idx, obs, epsilon, rng, safe=None):
    if agent.lookahead and len(idx):
        values = move_values(agent, snakes, idx, rng)
    else:
        values = agent.q_values(obs)
    values, safe = _apply_route_safety(agent, obs, values, safe)
    return agent.explore(values.argmax(1), epsilon, safe)


def _apply_route_safety(agent, obs, values, safe):
    if agent.shield and len(obs):
        allowed = route_safe(obs)
        limited = allowed.any(1)
        values = np.where(limited[:, None] & ~allowed, -np.inf, values)
        safe = allowed if safe is None else np.where(limited[:, None], allowed, safe)
    return values, safe


def solo_step(agent, snakes, obs, rng):
    idx = np.zeros(1, dtype=np.int64)
    sim = sim_obs = None
    if agent.lookahead:
        values, sim, sim_obs = try_moves(agent, snakes, idx, rng)
    else:
        values = agent.q_values(obs)
    values, _ = _apply_route_safety(agent, obs, values, None)
    a = int(values[0].argmax())
    if sim is not None and sim_obs is not None and sim.score[a] == snakes.score[0]:
        snakes.take(0, sim, a)
        return sim_obs[a:a + 1]
    snakes.step(np.array([a], dtype=np.int64))
    return observe(snakes, agent.senses)
