import numpy as np
import pytest

from snake import config as C
from snake.game import (Snakes, STRAIGHT, TURN_LEFT, TURN_RIGHT,
                        ALIVE, DIED_WALL, DIED_SELF, DIED_STARVED)
from snake.oracle import grade_moves
from snake.senses import observe

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def make(n=1, grid=10):
    s = Snakes(n, grid, np.random.default_rng(0))
    s.reset()
    return s


def act(*moves):
    return np.array(moves, dtype=np.int64)


def test_reset_spawns_legal_snakes_and_apples():
    s = make(n=200, grid=12)
    for i in range(s.n):
        body = s.body_mask(i)
        assert body.sum() == C.START_LENGTH
        assert not body[tuple(s.apple[i])]
    assert s.alive.all()


def test_each_snake_gets_its_own_seed():
    s = make(n=50)
    assert len(set(s.seeds.tolist())) == 50


def test_everyone_starts_in_the_same_place():
    s = make(n=20, grid=16)
    assert (s.head == (8, 8)).all() and (s.dir == UP).all()
    assert (s.stamp[:, 8:11, 8] == [0, -1, -2]).all()


def test_same_seed_same_apples_whatever_the_brain_does():
    # Two copies of one game: one snake eats its apple, the other wanders first.
    a, b = Snakes(1, 20), Snakes(1, 20)
    a.reset([123])
    b.reset([123])
    assert (a.apple == b.apple).all()
    first = a.apple[0].copy()

    def eat(s):
        # Teleport the head next to the apple and step onto it.
        y, x = s.apple[0]
        s.set_snake(0, [(y + 1, x), (y + 2, x), (y + 3, x)] if y + 3 < 20 else
                    [(y - 1, x), (y - 2, x), (y - 3, x)], UP if y + 3 < 20 else DOWN, (y, x))
        s.apples_placed[0] = 1
        s.step(act(STRAIGHT))

    b.step(act(TURN_LEFT))                   # b does something different first
    eat(a)
    eat(b)
    assert (a.apple == b.apple).all() and not (a.apple[0] == first).all()


def test_moving_straight():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 0))
    s.step(act(STRAIGHT))
    assert tuple(s.head[0]) == (4, 5)
    assert s.body_mask(0).sum() == 3
    assert not s.body_mask(0)[7, 5]          # tail moved on


def test_turns_are_relative():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 0))
    s.step(act(TURN_RIGHT))
    assert tuple(s.head[0]) == (5, 6) and s.dir[0] == RIGHT
    s.step(act(TURN_LEFT))
    assert tuple(s.head[0]) == (4, 6) and s.dir[0] == UP


def test_wall_kills():
    s = make()
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(9, 9))
    reward, done, _ = s.step(act(STRAIGHT))
    assert done[0] and not s.alive[0] and s.death[0] == DIED_WALL
    assert reward[0] == C.REWARD_DEATH


def test_self_collision_kills():
    s = make()
    # Head at (5,5) facing left, body curls round so turning right (up) hits (4,5).
    body = [(5, 5), (5, 6), (4, 6), (4, 5), (4, 4), (3, 4)]
    s.set_snake(0, body, LEFT, apple=(9, 9))
    _, done, _ = s.step(act(TURN_RIGHT))
    assert done[0] and s.death[0] == DIED_SELF


def test_chasing_own_tail_is_allowed():
    s = make()
    # A 2x2 loop: the head moves into the cell the tail is leaving.
    body = [(5, 5), (5, 6), (4, 6), (4, 5)]
    s.set_snake(0, body, LEFT, apple=(9, 9))
    _, done, _ = s.step(act(TURN_RIGHT))
    assert not done[0] and s.alive[0]


def test_eating_grows_and_moves_apple():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(4, 5))
    reward, _, ate = s.step(act(STRAIGHT))
    assert ate[0] and s.score[0] == 1 and s.length[0] == 4
    assert reward[0] == C.REWARD_APPLE
    assert tuple(s.apple[0]) != (4, 5)
    assert not s.body_mask(0)[tuple(s.apple[0])]


def test_starvation_is_strict():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(5, 7))
    budget = int(s.budget[0])
    assert budget == max(2 + C.STARVE_SLACK + C.STARVE_SLACK_PER_SEGMENT * 3,
                         int(C.STARVE_AREA_FACTOR * 10 * 10))
    # Spin in a 2x2 circle, never eating.
    moves = 0
    while s.alive[0]:
        reward, done, _ = s.step(act(TURN_RIGHT))
        moves += 1
    assert s.death[0] == DIED_STARVED and reward[0] == C.REWARD_STARVE
    assert moves == budget + 1


def test_snakes_do_not_interact():
    s = make(n=2)
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 0))
    s.set_snake(1, [(4, 4), (4, 5), (4, 6)], LEFT, apple=(0, 9))
    # Snake 0 moves straight into cells snake 1 occupies.
    _, done, _ = s.step(act(STRAIGHT, STRAIGHT))
    assert not done.any()


def test_dead_snakes_stay_dead():
    s = make(n=2)
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(9, 9))
    s.set_snake(1, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 0))
    s.step(act(STRAIGHT, STRAIGHT))
    head = tuple(s.head[0])
    _, done, _ = s.step(act(STRAIGHT, STRAIGHT))
    assert not done[0] and tuple(s.head[0]) == head


def test_senses():
    s = make()
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(5, 9))
    o = observe(s, "basic")[0]
    assert list(o[:3]) == [1, 0, 0]              # wall straight ahead only
    assert list(o[3:7]) == [1, 0, 0, 0]          # facing up
    assert list(o[7:]) == [0, 1, 0, 1]           # apple is down and right


# ---------------------------------------------------------------- referee

def test_oracle_prefers_shortest_path():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(5, 8))   # apple 3 to the right
    g = grade_moves(s)[0]
    assert g[TURN_RIGHT] == 1.0
    assert 0 < g[STRAIGHT] < 1.0
    assert 0 < g[TURN_LEFT] < g[STRAIGHT]


def test_oracle_diagonal_has_two_best_moves():
    s = make()
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(2, 8))   # up and right
    g = grade_moves(s)[0]
    assert g[STRAIGHT] == 1.0 and g[TURN_RIGHT] == 1.0


def test_oracle_fatal_move_scores_zero():
    s = make()
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(0, 9))
    g = grade_moves(s)[0]
    assert g[STRAIGHT] == 0.0
    assert g[TURN_RIGHT] == 1.0


def test_oracle_spots_trap():
    s = make(grid=10)
    # Head at (5,5) facing up. The body wraps round so that the cells straight ahead (4,5)
    # and to the right (5,6) are one-cell dead ends; only turning left escapes.
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    s.set_snake(0, body, UP, apple=(8, 4))
    g = grade_moves(s)[0]
    assert g[TURN_LEFT] == 1.0
    assert g[STRAIGHT] == pytest.approx(C.GRADE_TRAPPED)
    assert g[TURN_RIGHT] == pytest.approx(C.GRADE_TRAPPED)


def test_oracle_counts_tail_as_moving():
    s = make(grid=10)
    # Same wrap but short: the tail at (4,4) will have moved on by the time the head
    # gets there, so going straight is not a trap.
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4)]
    s.set_snake(0, body, UP, apple=(8, 4))
    g = grade_moves(s)[0]
    assert g[STRAIGHT] > C.GRADE_TRAPPED


def test_oracle_punishes_greedy_trap():
    s = make(grid=10)
    # Same wrap as the trap test, but the apple sits in the one-cell dead end straight
    # ahead. Eating it is possible, but then the snake is walled in by its own body and
    # dies. Turning left gives up the apple for now but keeps the tail reachable.
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    s.set_snake(0, body, UP, apple=(4, 5))
    g = grade_moves(s)[0]
    assert g[STRAIGHT] == pytest.approx(C.GRADE_GREEDY)   # eats, then boxed in
    assert g[TURN_LEFT] == 1.0                           # survives
    assert g[TURN_RIGHT] == pytest.approx(C.GRADE_TRAPPED)


def test_oracle_eats_when_safe():
    s = make(grid=20)
    s.set_snake(0, [(10, 10), (11, 10), (12, 10)], UP, apple=(5, 10))
    g = grade_moves(s)[0]
    assert g[STRAIGHT] == 1.0
    assert g[TURN_LEFT] < 1.0 and g[TURN_RIGHT] < 1.0


def test_apple_placement_is_fast_path_equivalent():
    # Same seeds on two boards -> same apples, even across many rounds.
    a, b = Snakes(300, 25), Snakes(300, 25)
    seeds = np.arange(300) * 7919
    a.reset(seeds)
    b.reset(seeds)
    assert (a.apple == b.apple).all()
    for i in range(300):
        assert not a.body_mask(i)[tuple(a.apple[i])]


# ---------------------------------------------------------------- vision senses

def vision(s):
    return observe(s, "vision")[0]


def test_vision_wall_and_open_moves():
    s = make(grid=10)
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(5, 9))
    v = vision(s)
    assert v[1 * 5] == 1.0                       # straight = into the wall
    for a in (0, 2):                             # left / right are open, tail reachable
        assert v[a * 5] == 0.0 and v[a * 5 + 1] > 0.9 and v[a * 5 + 3] == 1.0
    assert v[15] == 1.0                          # sight line ahead: wall is 1 step away


def test_vision_sees_dead_ends():
    s = make(grid=10)
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    s.set_snake(0, body, UP, apple=(8, 4))
    v = vision(s)
    straight, left = 5, 0
    assert v[straight + 1] < 0.02 and v[straight + 3] == 0.0 and v[straight + 4] == 0.0
    assert v[left + 1] > 0.5 and v[left + 3] == 1.0 and v[left + 4] > 0.0


def test_vision_knows_the_tail_moves_away():
    s = make(grid=10)
    # Short wrap: the would-be dead end opens up because the tail at (4,4) moves on in time.
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4)]
    s.set_snake(0, body, UP, apple=(8, 4))
    v = vision(s)
    assert v[5 + 1] > 0.5 and v[5 + 3] == 1.0


def test_vision_apple_position_is_relative():
    s = make(grid=10)
    s.set_snake(0, [(5, 5), (5, 4), (5, 3)], RIGHT, apple=(2, 5))   # facing right, apple 3 up
    v = vision(s)
    assert v[39] == pytest.approx(0.0)           # not ahead or behind
    assert v[40] == pytest.approx(-0.3)          # 3 to the left of the way it faces


def test_safe_moves_and_safe_exploration():
    from snake.agent import Agent
    s = make(n=1, grid=10)
    s.set_snake(0, [(0, 0), (1, 0), (2, 0)], UP, apple=(9, 9))     # top-left corner facing up
    assert s.safe_moves()[0].tolist() == [False, False, True]    # only turning right is safe
    agent = Agent(np.random.default_rng(0), 0, "basic")
    states = np.zeros((500, 11), dtype=np.float32)
    acts = agent.act(states, 1.0, np.repeat(s.safe_moves(), 500, axis=0))
    assert (acts == TURN_RIGHT).all()


# A snake whose body forms a closed ring with its head inside a 3x3 pocket.
_RING = [(4, 4), (4, 3), (4, 2), (5, 2), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 6), (4, 6),
         (3, 6), (2, 6), (2, 5), (2, 4), (2, 3), (2, 2), (3, 2)]
_OUTSIDE = [(3, 1), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0),
            (11, 1), (11, 2), (11, 3)]


def test_boxed_in_but_roomy_enough_to_wait_is_fine():
    s = make(grid=12)
    # Short tail outside the ring: the ring opens after 3 moves and the pocket has 7 free
    # cells to circle in, so the snake can wait it out. Not a trap.
    s.set_snake(0, _RING + _OUTSIDE[:2], RIGHT, apple=(11, 11))
    assert grade_moves(s)[0].max() == 1.0


def test_boxed_in_for_too_long_is_doomed():
    s = make(grid=12)
    # Long tail: the ring stays shut for 13 moves but the pocket only has room for ~7.
    s.set_snake(0, _RING + _OUTSIDE[:12], RIGHT, apple=(11, 11))
    assert grade_moves(s)[0].max() <= C.GRADE_TRAPPED


def test_filling_the_board_wins():
    s = make(grid=8)
    # Snake covers every square except the apple, in a serpentine; eating it fills the board.
    cells = []
    for y in range(8):
        row = [(y, x) for x in range(8)]
        cells += row if y % 2 == 0 else row[::-1]
    # cells runs from (0,0) to (7,0); the head is the last cell, apple the one before... use:
    body = cells[::-1][1:]            # head at (7,1), tail at (0,0); (7,0) is left for the apple
    s.set_snake(0, body, LEFT, apple=(7, 0))
    reward, done, ate = s.step(act(STRAIGHT))
    assert ate[0] and done[0] and s.death[0] == 4 and s.length[0] == 64
    assert reward[0] == C.REWARD_WIN


def test_shaping_fades_as_the_snake_grows():
    short, long_ = make(grid=10), make(grid=10)
    short.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 5))
    body = [(5, 5)] + [(6, x) for x in range(5, -1, -1)] + [(7, x) for x in range(10)]
    long_.set_snake(0, body, UP, apple=(0, 5))
    r_short, _, _ = short.step(act(STRAIGHT))
    r_long, _, _ = long_.step(act(STRAIGHT))
    assert 0 < r_long[0] < r_short[0] <= C.REWARD_CLOSER


def test_vision2_lookahead():
    s = make(grid=12)
    s.set_snake(0, _RING + _OUTSIDE[:2], RIGHT, apple=(11, 11))    # roomy pocket
    v = observe(s, "vision2")[0]
    assert v.shape == (51,)
    assert v[42 + 1 * 3] == 1.0                   # straight: can wait it out
    s.set_snake(0, _RING + _OUTSIDE[:12], RIGHT, apple=(11, 11))   # pocket shut too long
    v = observe(s, "vision2")[0]
    for a in range(3):
        assert v[42 + a * 3] == 0.0               # no escape whichever way
        assert v[42 + a * 3 + 1] > 0.0            # ...and it can see how long the wait is


def test_vision2_eat_safe():
    s = make(grid=10)
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    s.set_snake(0, body, UP, apple=(4, 5))        # apple in a one-cell dead end ahead
    v = observe(s, "vision2")[0]
    assert v[42 + 1 * 3 + 2] == 0.0               # eating it straight ahead is not safe
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(1, 5))
    v = observe(s, "vision2")[0]
    assert v[42 + 1 * 3 + 2] == 1.0 and v[42 + 1 * 3] == 1.0


# ---------------------------------------------------------------- compare + replays

def test_arena_waits_at_round_end_and_replays_deaths(tmp_path, monkeypatch):
    from snake import models
    from snake.agent import Agent
    from snake.arena import Arena
    path = tmp_path / "brain.pt"
    Agent(np.random.default_rng(0), 0, "basic").save(str(path))       # untrained: dies quickly
    monkeypatch.setattr(models, "brain_path", lambda name: str(path))
    arena = Arena(["a", "b"], 30, 12, seed=1)
    for _ in range(5000):
        if arena.tick():
            break
    assert arena.round_over and arena.round_no == 1
    assert not arena.tick() and arena.round_no == 1          # does not start the next round itself

    b = arena.boards[0]
    for i in range(arena.n):
        game = arena.replay(b, i)
        heads = game["heads"]
        assert len(heads) == b.death_move[i] + 1
        assert (np.abs(np.diff(heads, axis=0)).sum(1) == 1).all()   # one square per move
        assert game["died"]
        if game["cause"] in (1, 2):                                  # the fatal move is next door
            assert np.abs(np.array(game["death_cell"]) - heads[-1]).sum() == 1
    arena.new_round()
    assert arena.round_no == 2 and not arena.round_over


# ---------------------------------------------------------------- learning helpers

def test_nstep_sums_rewards_and_flushes_on_death():
    from snake.agent import NStep
    g = 0.9
    ns = NStep(1, 2, 3, g)
    idx = np.array([0])
    got = []
    for k, r in enumerate([1, 2, 3, 4, 5]):
        s = np.full((1, 2), k, dtype=np.float32)
        s2 = np.full((1, 2), k + 1, dtype=np.float32)
        out = ns.push(idx, s, np.array([k]), np.array([r], dtype=np.float32), s2, np.array([k == 4]))
        if out is not None:
            got += list(zip(out[1].tolist(), out[2].tolist(), out[4].tolist()))
    assert [a for a, _, _ in got] == [0, 1, 2, 3, 4]          # every move becomes one experience
    rets = dict((a, r) for a, r, _ in got)
    assert rets[0] == pytest.approx(1 + 2 * g + 3 * g * g)
    assert rets[2] == pytest.approx(3 + 4 * g + 5 * g * g)
    assert rets[3] == pytest.approx(4 + 5 * g)
    assert rets[4] == pytest.approx(5)
    assert [d for _, _, d in got] == [0, 0, 1, 1, 1]


def test_nstep_of_one_is_plain_q_learning():
    from snake.agent import NStep
    ns = NStep(2, 1, 1, 0.9)
    out = ns.push(np.array([0, 1]), np.zeros((2, 1), np.float32), np.array([1, 2]),
                  np.array([0.5, -1.0], np.float32), np.ones((2, 1), np.float32), np.array([False, True]))
    assert out[2].tolist() == [0.5, -1.0] and out[4].tolist() == [0.0, 1.0]


def test_fenced_off_measures_shut_off_space():
    from snake.senses import fenced_off
    s = make(grid=12)
    s.set_snake(0, _RING + _OUTSIDE[:2], RIGHT, apple=(11, 11))
    frac = fenced_off(s)[0]
    free = 144 - s.length[0]
    assert frac == pytest.approx((free - 7) / free)     # only the 7 pocket squares are reachable
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(0, 0))
    assert fenced_off(s)[0] == 0.0


def test_vision3_hug_and_fence():
    left, straight = 51, 53                     # (hug, fenced) pairs per move: left, straight, right
    s = make(grid=10)
    s.set_snake(0, [(5, 1), (6, 1), (7, 1)], UP, apple=(0, 9))    # one square away from the left wall
    v = observe(s, "vision3")[0]
    assert v.shape == (57,)
    assert v[straight] == 0.0                   # (4,1): nothing around it
    assert v[left] == pytest.approx(1 / 3)      # (5,0): against the wall
    assert v[left + 1] == 0.0 and v[straight + 1] == 0.0          # nothing gets fenced off
    s.set_snake(0, [(5, 0), (6, 0), (7, 0)], UP, apple=(0, 9))    # hugging the left wall
    v = observe(s, "vision3")[0]
    assert v[straight] == pytest.approx(1 / 3)                    # straight keeps hugging it
    s = make(grid=12)
    s.set_snake(0, _RING + _OUTSIDE[:2], RIGHT, apple=(11, 11))   # head inside its own ring
    v = observe(s, "vision3")[0]
    assert v[straight + 1] > 0.9                # nearly the whole board is shut off from it


def test_replay_lines_up_both_deaths(tmp_path, monkeypatch):
    from snake import models
    from snake.agent import Agent
    from snake.arena import Arena
    from snake.render import Replay
    for name, seed in (("a", 0), ("b", 1)):                     # two different (untrained) brains
        Agent(np.random.default_rng(seed), seed, "basic").save(str(tmp_path / f"{name}.pt"))
    monkeypatch.setattr(models, "brain_path", lambda name: str(tmp_path / f"{name}.pt"))
    arena = Arena(["a", "b"], 40, 12, seed=4)
    while not arena.tick():
        pass
    # Find a snake whose two games lasted different lengths.
    i = next(i for i in range(40) if arena.boards[0].death_move[i] != arena.boards[1].death_move[i])
    r = Replay(arena, i, moves_per_second=2)                  # 10 s at 2 moves/s = 20 moves back
    assert r.k == max(r.first, -20)
    r.k = 0                                                     # the last position before dying...
    assert [r.frame(b)[0] for b in range(2)] == r.moves         # ...is each game's own last move
    r.step(1)
    assert all(r.frame(b)[1] for b in range(2))                 # then both show their fatal move
    r.step(-5)
    assert [r.frame(b)[0] for b in range(2)] == [max(0, m - 4) for m in r.moves]


# ---------------------------------------------------------------- vision4, positions, look-ahead, stages

def test_vision4_tail_apple_and_split():
    s = make(grid=10)
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(1, 5))
    v = observe(s, "vision4")[0]
    assert v.shape == (66,)
    assert np.array_equal(v[:57], observe(s, "vision3")[0])
    straight = 57 + 3
    assert v[straight + 1] == 1.0                 # apple still reachable
    assert v[straight + 2] == 0.0                 # board still in one piece
    assert 0.0 < v[straight] <= 1.0               # tail reachable
    s = make(grid=12)
    s.set_snake(0, _RING + _OUTSIDE[:12], RIGHT, apple=(11, 11))   # head shut inside its ring
    v = observe(s, "vision4")[0]
    assert v[straight + 1] == 0.0                 # apple fenced off
    assert v[straight] == 0.0                     # tail out of reach
    assert v[straight + 2] > 0.0                  # the rest of the board is a separate piece


def test_snapshot_and_restore_round_trip():
    a = make(n=3, grid=12)
    a.set_snake(1, _RING + _OUTSIDE[:4], RIGHT, apple=(11, 11))
    a.score[1] = 20
    for _ in range(3):
        a.step(act(STRAIGHT, TURN_LEFT, STRAIGHT))
    snap = a.snapshot([1])
    b = make(n=2, grid=12)
    b.restore([0], snap, np.array([0]), np.array([99]))
    assert b.score[0] == a.score[1] and b.length[0] == a.length[1]
    assert (b.head[0] == a.head[1]).all() and b.dir[0] == a.dir[1]
    assert (b.body_mask(0) == a.body_mask(1)).all()
    assert b.alive[0] and not b.body_mask(0)[tuple(b.apple[0])]
    b.step(act(STRAIGHT, STRAIGHT))
    a.step(act(STRAIGHT, STRAIGHT, STRAIGHT))
    assert (b.head[0] == a.head[1]).all()        # plays on exactly like the original


def test_copies_are_independent_and_hide_the_real_apple():
    s = make(n=2, grid=10)
    s.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(4, 5))
    sim = Snakes.copies(s, np.array([0]), 3, np.array([1, 2, 3]))
    sim.step(np.array([0, 1, 2]))
    assert tuple(s.head[0]) == (5, 5)             # the original didn't move
    assert sim.score.tolist() == [0, 1, 0]        # only the straight copy ate
    real_next = make(n=1, grid=10)
    real_next.set_snake(0, [(5, 5), (6, 5), (7, 5)], UP, apple=(4, 5))
    real_next.seeds[0] = s.seeds[0]
    real_next.step(act(STRAIGHT))
    # The copy's next apple comes from its own seed, not the real game's.
    assert sim.seeds[1] != s.seeds[0]


def test_lookahead_avoids_the_trap_the_brain_cant_see():
    from snake.agent import Agent
    from snake.lookahead import choose_moves, move_values
    agent = Agent(np.random.default_rng(0), 0, "basic", lookahead=True)
    s = make(grid=10)
    body = [(5, 5), (6, 5), (6, 6), (6, 7), (5, 7), (4, 7), (4, 6), (3, 6), (3, 5),
            (3, 4), (4, 4), (4, 3), (4, 2), (4, 1)]
    s.set_snake(0, [(0, 5), (1, 5), (2, 5)], UP, apple=(9, 9))     # wall straight ahead
    v = move_values(agent, s, np.array([0]), np.random.default_rng(0))[0]
    assert v[STRAIGHT] == pytest.approx(C.REWARD_DEATH)           # it sees the crash coming
    acts = choose_moves(agent, s, np.array([0]), observe(s, "basic")[[0]], 0.0, np.random.default_rng(0))
    assert acts[0] != STRAIGHT


def test_stage_plan():
    from snake.trainer import stage_plan
    assert stage_plan(20, True) == [("short", 0, 50), ("mid", 50, 150), ("long", 150, None)]
    assert stage_plan(10, True) == [("short", 0, 50), ("long", 50, None)]    # 150 doesn't fit on 10x10
    assert stage_plan(20, False) == [("full", 0, None)]


# ---------------------------------------------------------------- the route (vision5)

@pytest.mark.parametrize("g", [8, 10, 12, 20])
def test_route_visits_every_square_once_in_a_loop(g):
    from snake.cycle import route, route_cells
    order = route(g)
    assert sorted(order.ravel().tolist()) == list(range(g * g))
    cells = route_cells(g)
    steps = np.abs(np.diff(np.vstack([cells, cells[:1]]), axis=0)).sum(1)
    assert (steps == 1).all()                              # each square next to the one after it
    n = g * g                                              # the starting body runs tail -> head
    assert (order[g // 2, g // 2] - order[g // 2 + 1, g // 2]) % n == 1
    assert (order[g // 2 + 1, g // 2] - order[g // 2 + 2, g // 2]) % n == 1


def test_odd_boards_have_no_route():
    from snake.cycle import route
    assert (route(9) == -1).all()
    s = make(grid=9)
    assert observe(s, "vision5")[0, 66:].sum() == 0.0


def test_vision5_at_the_start():
    s = make(grid=10)
    v = observe(s, "vision5")[0]
    assert v.shape == (81,)
    assert np.array_equal(v[:66], observe(s, "vision4")[0])
    assert v[80] == 1.0                                    # starting body is in route order
    follows = [a for a in range(3) if v[66 + a * 4 + 1] == 1.0]
    assert len(follows) == 1 and v[66 + follows[0] * 4 + 2] == 1.0   # following is safe


def test_route_senses_are_enough_to_fill_the_board():
    s = make(n=5, grid=8)
    s.reset(np.arange(5) + 11)
    area = 64
    for _ in range(5000):
        if not s.alive.any():
            break
        obs = observe(s, "vision5")
        acts = np.ones(5, dtype=np.int64)
        for i in np.flatnonzero(s.alive):
            o = obs[i]
            ok = [a for a in range(3) if o[a * 5] == 0 and o[66 + a * 4 + 2] == 1
                  and (s.length[i] <= area // 2 or o[66 + a * 4 + 1] == 1)]
            acts[i] = min(ok, key=lambda a: o[66 + a * 4 + 3])
        s.step(acts)
    assert (s.death == 4).all()                            # every game filled the board


@pytest.mark.parametrize("lookahead,shield", [(True, True), (True, False), (False, False)])
def test_fast_single_snake_engine_plays_exactly_like_normal_play(lookahead, shield):
    from snake.agent import Agent
    from snake.fast import FastPlayer
    from snake.lookahead import choose_moves
    agent = Agent(np.random.default_rng(3), 3, "vision5", lookahead=lookahead, shield=shield)
    a, b = Snakes(1, 10), Snakes(1, 10)
    a.reset(np.array([77]))
    b.reset(np.array([77]))
    ra, rb = np.random.default_rng(5), np.random.default_rng(5)
    fp = FastPlayer(agent, b, rb)
    oa, ob = observe(a, "vision5"), fp.observe()
    for move in range(400):
        if not a.alive[0]:
            break
        rb.bit_generator.state = ra.bit_generator.state      # same imagined apples in the look-ahead
        a.step(choose_moves(agent, a, np.array([0]), oa, 0.0, ra))
        oa = observe(a, "vision5")
        ob = fp.step(ob)
        assert (a.head == b.head).all() and a.score[0] == b.score[0] and a.alive[0] == b.alive[0], move
        assert (a.body_mask(0) == b.body_mask(0)).all() and (a.apple == b.apple).all(), move
