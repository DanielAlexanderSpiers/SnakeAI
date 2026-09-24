"""Look-ahead: try every move out before choosing one.

Without look-ahead the brain picks the move whose predicted future reward Q(state, move)
is highest, judging from the current position only. With look-ahead each snake plays all
three moves on copies of the board first, then scores each one as

    reward for that move  +  gamma * (the brain's best Q from where it ended up)

so it sees the actual position each move leads to, including the ones that box it in,
before it commits. It costs about 3x the work per move (3 copies to sense and judge).

The copies place any new apple with their own random seeds, so trying a move out never
reveals where the real next apple will appear.
"""

import numpy as np

from . import config as C
from .game import Snakes
from .senses import ROUTE_ORDER, fenced_off, observe


def move_values(agent, snakes, idx, rng):
    """(len(idx), 3): how good each move is for snakes idx, judged one move ahead."""
    return try_moves(agent, snakes, idx, rng)[0]


def try_moves(agent, snakes, idx, rng):
    """move_values, plus the copies that tried the moves and their senses (None if they all died)."""
    m = len(idx)
    sim = Snakes.copies(snakes, idx, 3, rng.integers(0, 2**62, size=3 * m))
    reward, done, _ = sim.step(np.tile(np.arange(3), m))
    value = reward.astype(np.float32)
    live = ~done
    obs = None
    if C.COMPACT_WEIGHT:
        # Same tidiness reward as in training. The part that depends on the current position
        # is the same for all three moves, so only the new position's share matters here.
        value -= C.COMPACT_WEIGHT * C.GAMMA * fenced_off(sim) * live
    if live.any():
        obs = observe(sim, agent.senses)
        if agent.senses == "vision5" and C.ROUTE_WEIGHT:
            value += C.ROUTE_WEIGHT * C.GAMMA * obs[:, ROUTE_ORDER] * live
        value[live] += C.GAMMA * agent.q_values(obs[live]).max(1)
    return value.reshape(m, 3), sim, obs


def route_safe(obs):
    """(n, 3) bool from vision5 senses: moves that keep the body in route order with room to
    spare, don't die, and don't skip past the apple along the route (so every one of them
    brings the apple closer). A snake that only ever makes such moves can neither crash nor
    starve, so it fills the board."""
    dies = obs[:, [0, 5, 10]] == 1.0
    safe = obs[:, [66 + 2, 70 + 2, 74 + 2]] == 1.0
    closer = obs[:, [66 + 3, 70 + 3, 74 + 3]] <= obs[:, [79]] + 1e-6
    return safe & closer & ~dies


def choose_moves(agent, snakes, idx, obs, epsilon, rng, safe=None):
    """Moves for snakes idx (obs = their current senses): brain alone, or brain + look-ahead.

    With route safety on, a snake that has at least one route-safe move may only pick among
    those (the brain still decides which one); otherwise it chooses freely."""
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
    """Play one move for the single snake in `snakes` (obs = its senses, shape (1, n)) as fast
    as possible. Returns its senses after the move.

    Same choice as choose_moves, but when look-ahead already played the chosen move on a
    copy and it didn't eat (eating is the one thing a copy does differently: it places its
    next apple with its own seed), the copy simply becomes the real snake instead of
    playing the move a second time.
    """
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
