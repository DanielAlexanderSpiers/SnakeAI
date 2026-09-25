import time

import numba
import numpy as np
import pygame
import torch

from . import models
from .agent import Agent
from .fast import FastPlayer
from .game import Snakes, DIED_SELF, DIED_WALL, DIED_WON
from .render import (ACCENT, APPLE, APPLE_EDGE, BG, CHECK_A, CHECK_B, DEATHS, DIM, PANEL, TEXT, WARN,
                     tail_shade)

SPEEDS = [10, 30, 100, 300, 1000, 3000, 10000, 0]
FRAME = 1 / 30
BODY = np.array([70, 220, 90], dtype=np.float32)
HEAD = np.array([170, 255, 180], dtype=np.float32)
PANEL_W = 340


class Solo:
    def __init__(self, name, grid, seed=None):
        numba.set_num_threads(1)
        torch.set_num_threads(1)
        self.name = name
        self.agent = Agent.from_file(models.brain_path(name))
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.snake = Snakes(1, grid, np.random.default_rng(self.rng.integers(2**62)))
        self.player = FastPlayer(self.agent, self.snake, self.rng)
        self.scores, self.moves, self.fill_moves, self.causes = [], [], [], []
        self.last = None
        self.final = None
        self.new_game()

    def new_game(self):
        self.snake.reset()
        self.obs = self.player.observe()
        self.play_time = 0.0

    def play(self, budget):
        s, step = self.snake, self.player.step
        started = time.perf_counter()
        deadline = started + budget if isinstance(budget, float) else None
        limit = budget if isinstance(budget, int) else None
        played, filled = 0, False
        obs = self.obs
        while True:
            obs = step(obs)
            played += 1
            if not s.alive[0]:
                self.obs = obs
                self.play_time += time.perf_counter() - started
                started = time.perf_counter()
                filled = self._game_over()
                obs = self.obs
                if filled:
                    break
            if limit is not None:
                if played >= limit:
                    break
            elif played % 64 == 0 and time.perf_counter() >= deadline:
                break
        self.obs = obs
        self.play_time += time.perf_counter() - started
        return played, filled

    def _game_over(self):
        s = self.snake
        score, moves, cause = int(s.score[0]), int(s.t[0]), int(s.death[0])
        if cause in (DIED_WALL, DIED_SELF):
            moves += 1
        self.scores.append(score)
        self.moves.append(moves)
        self.causes.append(cause)
        self.last = (score, moves, cause)
        won = cause == DIED_WON
        if won:
            self.fill_moves.append(moves)
            self.final = dict(stamp=s.stamp[0].copy(), t=int(s.t[0]), length=int(s.length[0]),
                              score=score, moves=moves, seconds=self.play_time, game=len(self.scores))
        self.new_game()
        return won


class SoloWindow:
    def __init__(self, solo):
        self.solo = solo
        info = pygame.display.Info()
        size = min(info.current_h - 120, info.current_w - PANEL_W - 60)
        self.cell = max(3, size // solo.grid)
        self.board = self.cell * solo.grid
        self.height = max(self.board + 28, 620)
        self.screen = pygame.display.set_mode((self.board + 28 + PANEL_W, self.height))
        pygame.display.set_caption(f"SnakeAI - {solo.name} solo")
        self.font = pygame.font.SysFont("segoeui,arial", 16)
        self.bold = pygame.font.SysFont("segoeui,arial", 16, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 14)
        self.big = pygame.font.SysFont("segoeui,arial", 24, bold=True)
        self.huge = pygame.font.SysFont("segoeui,arial", 44, bold=True)
        g = solo.grid
        yy, xx = np.indices((g, g))
        self.base = np.where(((yy + xx) % 2).T[..., None], CHECK_B, CHECK_A).astype(np.uint8)
        self.shade = pygame.Surface((self.board, self.board), pygame.SRCALPHA)
        self.shade.fill((10, 12, 16, 175))

    def text(self, txt, x, y, font=None, colour=TEXT):
        surf = (font or self.font).render(txt, True, colour)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 1

    def draw_snake(self, stamp, t, length):
        img = self.base.copy()
        ys, xs = np.nonzero(stamp > t - length)
        age = t - stamp[ys, xs]
        cols = np.where((age == 0)[:, None], HEAD, BODY)
        img[xs, ys] = (cols * tail_shade(age, np.full(len(age), length))[:, None]).astype(np.uint8)
        surf = pygame.transform.scale(pygame.surfarray.make_surface(img), (self.board, self.board))
        self.screen.blit(surf, (14, 14))

    def draw(self, speed_label, tps, paused, end_screen=False):
        s = self.solo.snake
        self.screen.fill(BG)
        if end_screen and self.solo.final:
            f = self.solo.final
            self.draw_snake(f["stamp"], f["t"], f["length"])
            self.draw_end_screen(f)
        else:
            self.draw_snake(s.stamp[0], int(s.t[0]), int(s.length[0]))
            c = self.cell
            inset = c // 5 if c >= 6 else 0
            ay, ax = s.apple[0]
            rect = (14 + ax * c + inset, 14 + ay * c + inset, c - 2 * inset, c - 2 * inset)
            self.screen.fill(APPLE, rect)
            if c - 2 * inset >= 8:
                pygame.draw.rect(self.screen, APPLE_EDGE, rect, 1)
        self.draw_panel(speed_label, tps, paused, end_screen)
        pygame.display.flip()

    def draw_end_screen(self, f):
        solo = self.solo
        self.screen.blit(self.shade, (14, 14))
        n, filled = len(solo.scores), len(solo.fill_moves)
        lines = [
            (self.huge, "BOARD FILLED!", WARN),
            (self.big, f"{f['score']} apples in {f['moves']:,} moves", TEXT),
            (self.font, f"{f['seconds']:.1f} seconds of play   "
                        f"({f['moves'] / max(f['score'], 1):.0f} moves per apple)", TEXT),
            (self.font, f"game {f['game']}   boards filled so far: {filled} of {n}", DIM),
            (self.font, " ", DIM),
            (self.bold, "Space or N: next game        Esc: back to the menu", ACCENT),
        ]
        surfs = [font.render(txt, True, col) for font, txt, col in lines]
        y = 14 + (self.board - sum(sf.get_height() + 10 for sf in surfs)) // 2
        for sf in surfs:
            self.screen.blit(sf, (14 + (self.board - sf.get_width()) // 2, y))
            y += sf.get_height() + 10

    def draw_panel(self, speed_label, tps, paused, end_screen):
        solo, s = self.solo, self.solo.snake
        x0 = self.board + 28
        self.screen.fill(PANEL, (x0, 0, PANEL_W, self.height))
        x, y = x0 + 20, 14
        a = solo.agent
        extra = (" + look-ahead" if a.lookahead else "") + (" + route safety" if a.shield else "")
        y = self.text(solo.name, x, y, self.big)
        y = self.text(f"{a.senses}{extra}", x, y, self.small, DIM)
        y = self.text(f"one snake, {solo.grid}x{solo.grid} board", x, y, self.small, DIM) + 8
        if end_screen:
            y = self.text("BOARD FILLED - waiting for you", x, y, colour=WARN)
        else:
            y = self.text("PAUSED" if paused else f"Speed {speed_label}", x, y, colour=ACCENT if paused else TEXT)
        y = self.text(f"{tps:,.0f} moves per second", x, y, colour=DIM) + 10

        if not end_screen:
            area = solo.grid * solo.grid - 3
            y = self.text(f"Game {len(solo.scores) + 1}", x, y, self.bold)
            y = self.text(f"score {int(s.score[0])} of {area}   ({s.score[0] / area:.0%} of the board)", x, y)
            y = self.text(f"moves {int(s.t[0]):,}", x, y) + 10

        if solo.last:
            score, moves, cause = solo.last
            colour = WARN if cause == DIED_WON else TEXT
            y = self.text("Last game", x, y, self.bold)
            y = self.text(DEATHS.get(cause, "ended"), x, y, colour=colour)
            y = self.text(f"{score} apples in {moves:,} moves", x, y) + 10

        n = len(solo.scores)
        y = self.text(f"All {n} finished game{'s' if n != 1 else ''}", x, y, self.bold)
        if n:
            sc = np.array(solo.scores)
            filled = len(solo.fill_moves)
            y = self.text(f"average {sc.mean():.1f}   best {sc.max()}   median {np.median(sc):.0f}", x, y)
            y = self.text(f"boards filled {filled} of {n} ({filled / n:.0%})", x, y)
            if filled:
                fm = np.array(solo.fill_moves)
                y = self.text(f"moves per full board: avg {fm.mean():,.0f}", x, y)
                y = self.text(f"fastest {fm.min():,}   slowest {fm.max():,}", x, y, self.small, DIM)
            causes = np.bincount(solo.causes, minlength=5)
            y = self.text(f"wall {causes[1]}   itself {causes[2]}   starved {causes[3]}", x, y, self.small, DIM)
        else:
            y = self.text("results after the first game", x, y, self.small, DIM)
        y += 16
        for k in ("Space  pause", "Up / Down  speed", "M  maximum speed", "N  skip to a new game",
                  "Filling the board stops on an end screen", "Esc  back to menu"):
            y = self.text(k, x, y, self.small, DIM)


def run_solo(name, grid, seed=None):
    pygame.init()
    solo = Solo(name, grid, seed)
    window = SoloWindow(solo)
    speed = len(SPEEDS) - 1
    paused = False
    end_screen = False
    clock = pygame.time.Clock()
    owed = 0.0
    tps, count, since = 0.0, 0, time.perf_counter()
    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type != pygame.KEYDOWN:
                    continue
                elif e.key == pygame.K_ESCAPE:
                    running = False
                elif end_screen:
                    if e.key in (pygame.K_SPACE, pygame.K_n, pygame.K_RETURN):
                        end_screen = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(speed + 1, len(SPEEDS) - 1)
                elif e.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed - 1, 0)
                elif e.key == pygame.K_m:
                    speed = len(SPEEDS) - 1
                elif e.key == pygame.K_n:
                    solo.new_game()

            target = SPEEDS[speed]
            if not paused and not end_screen:
                if target == 0:
                    played, filled = solo.play(FRAME)
                else:
                    owed += target * FRAME
                    played, filled = solo.play(int(owed)) if owed >= 1 else (0, False)
                    owed -= played
                count += played
                if filled:
                    end_screen, owed = True, 0.0
            now = time.perf_counter()
            if now - since >= 0.5:
                tps, count, since = count / (now - since), 0, now
            window.draw("maximum" if target == 0 else f"{target:,} moves/s", tps, paused, end_screen)
            clock.tick(0 if target == 0 else 1 / FRAME)
    finally:
        pygame.quit()
    return solo
