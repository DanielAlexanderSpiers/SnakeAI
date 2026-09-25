import secrets
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from . import models
from .cpu import limit_threads
from .agent import Agent
from .game import Snakes, DELTAS, ACTION_TURN
from .lookahead import choose_moves
from .oracle import grade_moves
from .senses import observe


@dataclass
class Frame:
    idx: np.ndarray
    head: np.ndarray
    apple: np.ndarray
    length: np.ndarray
    grade: np.ndarray
    n_best: np.ndarray


@dataclass
class Board:
    name: str
    agent: Agent
    snakes: Snakes
    quality: np.ndarray
    last_grade: np.ndarray
    last_n_best: np.ndarray
    obs: np.ndarray = None
    grade_sum: float = 0.0
    grade_count: int = 0
    frames: list = field(default_factory=list)
    death_move: np.ndarray = None
    death_cell: np.ndarray = None
    round_avgs: list = field(default_factory=list)
    round_accs: list = field(default_factory=list)
    round_wins: float = 0.0
    seed_wins: int = 0
    seeds_played: int = 0
    boards_filled: int = 0
    scores: list = field(default_factory=list)
    fill_moves: list = field(default_factory=list)

    @property
    def fill_speed(self):
        return float(np.mean(self.fill_moves)) if self.fill_moves else 0.0

    @property
    def best(self):
        return max(self.scores) if self.scores else 0

    @property
    def median(self):
        return float(np.median(self.scores)) if self.scores else 0.0

    @property
    def accuracy(self):
        return self.grade_sum / self.grade_count if self.grade_count else 0.0


class Arena:
    def __init__(self, names, snakes, grid, seed=None):
        limit_threads()
        self.seed = int(seed) if seed is not None else secrets.randbits(62)
        self.rng = np.random.default_rng(self.seed)
        self.plan_rng = np.random.default_rng(self.seed + 1)
        self.n, self.grid = snakes, grid
        self.boards = []
        for name in names:
            agent = Agent.from_file(models.brain_path(name))
            self.boards.append(Board(name, agent, Snakes(snakes, grid),
                                     np.full(snakes, 0.5, dtype=np.float32),
                                     np.full(snakes, -1.0, dtype=np.float32),
                                     np.zeros(snakes, dtype=np.int64)))
        self.round_no = 0
        self.ticks = 0
        self.round_over = False
        self.new_round()

    def new_round(self):
        seeds = self.rng.integers(0, 2**62, size=self.n)
        for b in self.boards:
            b.snakes.reset(seeds)
            b.obs = observe(b.snakes, b.agent.senses)
            b.quality.fill(0.5)
            b.last_grade.fill(-1.0)
            b.last_n_best.fill(0)
            b.grade_sum, b.grade_count = 0.0, 0
            b.death_move = np.full(self.n, -1, dtype=np.int64)
            b.death_cell = np.zeros((self.n, 2), dtype=np.int64)
            b.frames = []
            self._record(b, np.arange(self.n), np.full(self.n, -1.0), np.zeros(self.n, dtype=np.int64))
        self.round_no += 1
        self.ticks = 0
        self.round_over = False

    def tick(self):
        if self.round_over:
            return False
        for b in self.boards:
            s = b.snakes
            if not s.alive.any():
                continue
            alive = np.flatnonzero(s.alive)
            acts = choose_moves(b.agent, s, alive, b.obs[alive], 0.0, self.plan_rng)
            actions = np.zeros(s.n, dtype=np.int64)
            actions[alive] = acts
            all_grades = grade_moves(s)[alive]
            grades = all_grades[np.arange(len(alive)), acts]
            n_best = (all_grades >= 0.999).sum(1)
            b.quality[alive] += C.QUALITY_EMA * (grades - b.quality[alive])
            b.last_grade[alive] = grades
            b.last_n_best[alive] = n_best
            b.grade_sum += float(grades.sum())
            b.grade_count += len(alive)
            target = s.head[alive] + DELTAS[(s.dir[alive] + ACTION_TURN[acts]) % 4]

            _, done, _ = s.step(actions)
            b.obs = observe(s, b.agent.senses)

            died = done[alive]
            b.death_move[alive[died]] = len(b.frames) - 1
            b.death_cell[alive[died]] = target[died]
            still = ~died
            self._record(b, alive[still], grades[still], n_best[still])
        self.ticks += 1
        if any(b.snakes.alive.any() for b in self.boards):
            return False
        self._finish_round()
        self.round_over = True
        return True

    @staticmethod
    def _record(b, idx, grades, n_best):
        s = b.snakes
        b.frames.append(Frame(idx.astype(np.int32), s.head[idx].astype(np.uint8),
                              s.apple[idx].astype(np.uint8), s.length[idx].astype(np.uint16),
                              np.asarray(grades, dtype=np.float32), np.asarray(n_best, dtype=np.uint8)))

    def _finish_round(self):
        avgs = [float(b.snakes.score.mean()) for b in self.boards]
        top = max(avgs)
        winners = [b for b, a in zip(self.boards, avgs) if a == top]
        scores = np.stack([b.snakes.score for b in self.boards])
        best_per_seed = scores.max(0)
        sole_best = (scores == best_per_seed).sum(0) == 1
        for k, (b, a) in enumerate(zip(self.boards, avgs)):
            b.round_avgs.append(a)
            b.round_accs.append(b.accuracy)
            if b in winners:
                b.round_wins += 1 / len(winners)
            b.seed_wins += int(((scores[k] == best_per_seed) & sole_best).sum())
            b.seeds_played += self.n
            b.boards_filled += int((b.snakes.death == 4).sum())
            b.scores.extend(b.snakes.score.tolist())
            filled = b.snakes.death == 4
            b.fill_moves.extend((b.death_move[filled] + 1).tolist())


    def replay(self, board, i):
        heads, apples, lengths, grades, n_best = [], [], [], [], []
        for f in board.frames:
            j = np.searchsorted(f.idx, i)
            if j >= len(f.idx) or f.idx[j] != i:
                break
            heads.append(f.head[j])
            apples.append(f.apple[j])
            lengths.append(f.length[j])
            grades.append(f.grade[j])
            n_best.append(f.n_best[j])
        g = self.grid
        start_tail = [(g // 2 + k, g // 2) for k in range(C.START_LENGTH - 1, 0, -1)]
        return dict(
            heads=np.array(heads, dtype=np.int64), apples=np.array(apples, dtype=np.int64),
            lengths=np.array(lengths, dtype=np.int64), grades=np.array(grades, dtype=np.float32),
            n_best=np.array(n_best, dtype=np.int64), start_tail=start_tail,
            died=int(board.death_move[i]) >= 0, cause=int(board.snakes.death[i]),
            death_cell=tuple(int(v) for v in board.death_cell[i]))
