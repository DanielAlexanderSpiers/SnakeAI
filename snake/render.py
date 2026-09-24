"""The compare window: one board per model side by side, plus a scoreboard panel.

Live mode plays the round. Follow mode (Tab) fades every snake except one, and since
snake #N has the same apples on every board you can watch the models handle exactly
the same game. When that snake has died everywhere the window pauses, and R replays
its game move by move up to (and including) the move that killed it.
"""

import time

import numpy as np
import pygame

from . import config as C
from .arena import Arena

BG = (15, 16, 21)
PANEL = (22, 24, 31)
CHECK_A = (26, 28, 36)
CHECK_B = (30, 32, 41)
TEXT = (225, 228, 235)
DIM = (130, 136, 150)
ACCENT = (110, 200, 255)
WARN = (255, 200, 90)
APPLE = (230, 30, 40)
APPLE_EDGE = (110, 0, 10)
MODEL_COLOURS = [(110, 200, 255), (255, 170, 60), (200, 150, 255), (235, 235, 235)]
FADED = np.array([48, 52, 62], dtype=np.uint8)       # other snakes while following one
GAP = 14
LABEL = 46
DEATHS = {1: "hit the wall", 2: "hit itself", 3: "starved", 4: "FILLED THE BOARD"}
TAIL_DARKEN = 0.55                                   # the tail end is drawn this much darker than the head
REPLAY_SECONDS = 10                                  # replays show the last 10 seconds of each life
REPLAY_SPEED = 20                                    # moves per second when the live speed is "fastest"


def quality_colours(q):
    """Quality 0..1 -> (body, head) RGB arrays. Red (bad) through yellow to green (good)."""
    v = np.clip(q, 0, 1) ** C.COLOUR_POWER
    r = np.where(v < 0.5, 1.0, 2.0 * (1.0 - v))
    g = np.where(v < 0.5, 2.0 * v, 1.0)
    rgb = np.stack([r, g, np.zeros_like(r)], axis=1)
    body = (40 + rgb * 190).astype(np.uint8)
    head = (90 + rgb * 165).astype(np.uint8)
    return body, head


def tail_shade(age, length):
    """Brightness for body cells: 1 at the head fading to 1 - TAIL_DARKEN at the tail,
    so you can follow the body round and see which end is which. age 0 = head."""
    span = np.maximum(1, length - 1).astype(np.float32)
    return 1.0 - TAIL_DARKEN * (age.astype(np.float32) / span)


def describe_grade(g, n_best=1):
    if g < 0:
        return "-"
    if g == 0:
        return "fatal move"
    if g >= 0.999:
        return f"best move (1 of {n_best} tied)" if n_best > 1 else "best move"
    if g > C.GRADE_STALL + 0.01:
        return "slower path"
    if abs(g - C.GRADE_STALL) < 1e-3:
        return "stalling (safe path to apple ignored)"
    if abs(g - C.GRADE_GREEDY) < 1e-3:
        return "greedy (eats, then can't escape)"
    return "doomed (boxed in)"


class Replay:
    """The followed snake's final moments on every board, lined up so that every
    board's snake dies at the same moment.

    The clock k counts moves relative to the deaths: k = 0 is each snake's last position
    and k = 1 shows the fatal move. It starts REPLAY_SECONDS before the deaths at the
    replay speed. A snake that is still alive is lined up by its latest position.
    """

    def __init__(self, arena, i, moves_per_second):
        self.i = i
        self.games = [arena.replay(b, i) for b in arena.boards]
        self.moves = [len(g["heads"]) - 1 for g in self.games]
        self.lead = max(1, int(round(REPLAY_SECONDS * moves_per_second)))
        self.first = -max(self.moves)                       # the very start of the longest game
        self.end = 1 if any(g["died"] for g in self.games) else 0
        self.k = max(self.first, -self.lead)
        self.playing = True
        self.owed = 0.0
        # Colour along the way, rebuilt from the recorded grades.
        self.quality = []
        for g in self.games:
            q = np.empty(len(g["grades"]), dtype=np.float32)
            v = 0.5
            for j, gr in enumerate(g["grades"]):
                if gr >= 0:
                    v += C.QUALITY_EMA * (gr - v)
                q[j] = v
            self.quality.append(q)

    def restart(self):
        self.k = max(self.first, -self.lead)

    def step(self, n):
        self.k = int(np.clip(self.k + n, self.first, self.end))
        if self.k == self.end:
            self.playing = False

    def frame(self, b):
        """(move number to show on board b, whether to show its fatal move)."""
        m = self.moves[b]
        return int(np.clip(m + min(self.k, 0), 0, m)), self.k > 0 and self.games[b]["died"]

    def seconds_left(self, moves_per_second):
        return max(0.0, -self.k) / moves_per_second


class Window:
    def __init__(self, arena):
        self.arena = arena
        k, grid = len(arena.boards), arena.grid
        self.cols = 1 if k == 1 else 2
        self.rows = 1 if k <= 2 else 2
        info = pygame.display.Info()
        avail_w = min(info.current_w - 40, 1900) - C.PANEL_WIDTH - GAP * (self.cols + 1)
        avail_h = info.current_h - 110 - GAP * (self.rows + 1) - LABEL * self.rows
        self.cell = max(2, min(avail_w // self.cols, avail_h // self.rows) // grid)
        self.board_px = self.cell * grid
        self.boards_w = self.cols * (self.board_px + GAP) + GAP
        width = self.boards_w + C.PANEL_WIDTH
        self.height = max(self.rows * (self.board_px + LABEL + GAP) + GAP, 720)
        self.screen = pygame.display.set_mode((width, self.height))
        pygame.display.set_caption("SnakeAI - compare models")
        self.font = pygame.font.SysFont("segoeui,arial", 16)
        self.bold = pygame.font.SysFont("segoeui,arial", 16, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial", 14)
        self.big = pygame.font.SysFont("segoeui,arial", 24, bold=True)
        yy, xx = np.indices((grid, grid))
        checker = ((yy + xx) % 2).T[..., None]            # surfarray wants [x, y]
        self.base = np.where(checker, CHECK_B, CHECK_A).astype(np.uint8)
        self.shade = pygame.Surface((self.board_px, self.board_px), pygame.SRCALPHA)
        self.shade.fill((10, 10, 14, 190))
        self.focus = None          # index of the followed snake (same seed on every board)
        self.focus_death_seen = False
        self.replay = None
        self.replay_mps = REPLAY_SPEED
        self.notice = ""

    # ------------------------------------------------------------ following

    def follow(self, step):
        """step 0: toggle following the top-scoring living snake; +1/-1: next/previous snake."""
        boards = self.arena.boards
        n = self.arena.n
        alive_any = np.zeros(n, dtype=bool)
        for b in boards:
            alive_any |= b.snakes.alive
        self.focus_death_seen = False
        if step == 0:
            if self.focus is not None:
                self.focus = None
                return
            score = np.max([np.where(b.snakes.alive | ~alive_any, b.snakes.score, -1) for b in boards], axis=0)
            self.focus = int(np.argmax(score))
            return
        start = 0 if self.focus is None else self.focus
        for k in range(1, n + 1):
            i = (start + step * k) % n
            if alive_any[i] or not alive_any.any():
                self.focus = i
                return

    def focus_dead_everywhere(self):
        return self.focus is not None and not any(b.snakes.alive[self.focus] for b in self.arena.boards)

    def start_replay(self, moves_per_second):
        if self.focus is None:
            # Nobody followed yet: take the snake that scored highest on any board.
            score = np.max([b.snakes.score for b in self.arena.boards], axis=0)
            self.focus = int(np.argmax(score))
        self.replay = Replay(self.arena, self.focus, moves_per_second)

    # ------------------------------------------------------------ drawing

    def text(self, txt, x, y, font=None, colour=TEXT):
        surf = (font or self.font).render(txt, True, colour)
        self.screen.blit(surf, (x, y))
        return y + surf.get_height() + 1

    def draw(self, speed_label, tps, paused):
        self.screen.fill(BG)
        for i, b in enumerate(self.arena.boards):
            col, row = i % self.cols, i // self.cols
            x = GAP + col * (self.board_px + GAP)
            y = GAP + row * (self.board_px + LABEL + GAP)
            if self.replay:
                self.draw_replay_board(i, x, y)
            else:
                self.draw_label(b, i, x, y)
                self.draw_board(b, x, y + LABEL)
        self.draw_panel(speed_label, tps, paused)
        if self.notice:
            self.draw_notice()
        pygame.display.flip()

    def draw_notice(self):
        surf = self.font.render(self.notice, True, BG)
        if surf.get_width() + 24 > self.screen.get_width() - 2 * GAP:
            surf = self.small.render(self.notice, True, BG)
        w = min(surf.get_width() + 24, self.screen.get_width() - 2 * GAP)
        rect = pygame.Rect(GAP, self.height - 44, w, 32)
        pygame.draw.rect(self.screen, WARN, rect, border_radius=6)
        self.screen.blit(surf, (rect.x + 12, rect.y + 6))

    def draw_label(self, b, i, x, y):
        s = b.snakes
        alive = int(s.alive.sum())
        self.screen.set_clip((x, y, self.board_px, LABEL))
        self.text(b.name, x, y, self.bold, MODEL_COLOURS[i])
        if self.focus is None:
            line = (f"alive {alive}/{s.n}   avg {s.score.mean():.2f}   best {int(s.score.max())}   "
                    f"acc {b.accuracy * 100:.1f}%")
        else:
            f = self.focus
            if s.alive[f]:
                state = "last move: " + describe_grade(b.last_grade[f], int(b.last_n_best[f]))
            else:
                state = DEATHS.get(int(s.death[f]), "dead")
            line = f"snake #{f + 1}   score {int(s.score[f])}   {state}"
        self.text(line, x, y + 21, self.small, DIM if self.focus is None else TEXT)
        self.screen.set_clip(None)

    def blit_board(self, img, x0, y0):
        surf = pygame.transform.scale(pygame.surfarray.make_surface(img), (self.board_px, self.board_px))
        self.screen.blit(surf, (x0, y0))

    def draw_apple(self, x0, y0, y, x):
        c = self.cell
        inset = c // 5 if c >= 6 else 0
        size = c - 2 * inset
        rect = (x0 + x * c + inset, y0 + y * c + inset, size, size)
        self.screen.fill(APPLE, rect)
        if size >= 8:
            pygame.draw.rect(self.screen, APPLE_EDGE, rect, 1)

    def draw_board(self, b, x0, y0):
        s = b.snakes
        img = self.base.copy()
        live = s.alive
        if live.any():
            li = np.flatnonzero(live)
            occ = s.stamp[li] > (s.t[li] - s.length[li])[:, None, None]
            k, ys, xs = np.nonzero(occ)
            n_idx = li[k]
            body, head = quality_colours(b.quality)
            if self.focus is not None:                              # followed snake on top, others faded
                key = (n_idx == self.focus).astype(np.float32)
                others = np.arange(s.n) != self.focus
                body, head = body.copy(), head.copy()
                body[others] = FADED
                head[others] = FADED
            else:
                key = b.quality[n_idx]                              # best snakes drawn on top
            order = np.argsort(key, kind="stable")
            n_idx, ys, xs = n_idx[order], ys[order], xs[order]
            age = s.t[n_idx] - s.stamp[n_idx, ys, xs]              # 0 = head, length-1 = tail
            cols = np.where((age == 0)[:, None], head[n_idx], body[n_idx]).astype(np.float32)
            shade = tail_shade(age, s.length[n_idx])
            if self.focus is not None:
                shade[n_idx != self.focus] = 1.0                   # faded snakes stay flat grey
            img[xs, ys] = (cols * shade[:, None]).astype(np.uint8)
        self.blit_board(img, x0, y0)

        shown = live.copy()
        if self.focus is not None:
            shown[:] = False
            shown[self.focus] = live[self.focus]
        for y, x in s.apple[shown]:
            self.draw_apple(x0, y0, y, x)

        if not live.any():
            self.screen.blit(self.shade, (x0, y0))
            msg = self.big.render(f"Finished  -  avg {s.score.mean():.2f}   best {int(s.score.max())}",
                                  True, TEXT)
            self.screen.blit(msg, (x0 + (self.board_px - msg.get_width()) // 2,
                                   y0 + self.board_px // 2 - msg.get_height() // 2))

    def draw_replay_board(self, bi, x0, y):
        r = self.replay
        game, moves = r.games[bi], r.moves[bi]
        b = self.arena.boards[bi]
        k, fatal = r.frame(bi)
        heads = game["heads"]
        L = int(game["lengths"][k])
        if k + 1 >= L:
            cells = heads[k + 1 - L:k + 1]
        else:                                  # still partly the body it started with
            cells = np.concatenate([np.array(game["start_tail"][-(L - k - 1):]), heads[:k + 1]])

        # Label
        self.screen.set_clip((x0, y, self.board_px, LABEL))
        self.text(f"{b.name}   REPLAY snake #{r.i + 1}", x0, y, self.bold, MODEL_COLOURS[bi])
        score = L - C.START_LENGTH
        if fatal:
            state = f"move {moves + 1}: {DEATHS.get(game['cause'], 'died')}"
        elif k == moves and not game["died"] and b.snakes.alive[r.i]:
            state = "still alive (its latest move)"
        else:
            state = "move {}: {}".format(k, describe_grade(game["grades"][k], int(game["n_best"][k])))
        self.text(f"score {score}   {state}", x0, y + 21, self.small, WARN if fatal else TEXT)
        self.screen.set_clip(None)

        # Board
        y0 = y + LABEL
        img = self.base.copy()
        body, head = quality_colours(np.array([r.quality[bi][k]]))
        age = np.arange(len(cells) - 1, -1, -1)                    # cells run tail -> head
        cols = np.repeat(body.astype(np.float32), len(cells), axis=0)
        cols[-1] = head[0]
        img[cells[:, 1], cells[:, 0]] = (cols * tail_shade(age, np.full(len(cells), len(cells)))[:, None]).astype(np.uint8)
        self.blit_board(img, x0, y0)
        ay, ax = game["apples"][k]
        if not fatal or game["cause"] == 3:
            self.draw_apple(x0, y0, ay, ax)
        if fatal:
            self.draw_death(x0, y0, game)

    def draw_death(self, x0, y0, game):
        c = self.cell
        dy, dx = game["death_cell"]
        cx = x0 + dx * c + c // 2
        cy = y0 + dy * c + c // 2
        cx = min(max(cx, x0), x0 + self.board_px)
        cy = min(max(cy, y0), y0 + self.board_px)
        s = max(4, c // 2)
        w = max(2, c // 6)
        pygame.draw.line(self.screen, (255, 255, 255), (cx - s, cy - s), (cx + s, cy + s), w)
        pygame.draw.line(self.screen, (255, 255, 255), (cx - s, cy + s), (cx + s, cy - s), w)

    # ------------------------------------------------------------ panel

    def draw_panel(self, speed_label, tps, paused):
        a = self.arena
        x0 = self.boards_w
        w = C.PANEL_WIDTH - 40
        self.screen.fill(PANEL, (x0, 0, C.PANEL_WIDTH, self.height))
        x, y = x0 + 20, 14
        y = self.text("Compare models", x, y, self.big)
        y = self.text(f"{a.n} snakes each, same apples for every model", x, y, self.small, DIM)
        y = self.text(f"seed {a.seed}", x, y, self.small, DIM) + 6
        status = "ROUND FINISHED" if a.round_over else f"tick {a.ticks:,}"
        y = self.text(f"Round {a.round_no}   {status}", x, y, colour=WARN if a.round_over else TEXT)
        if self.replay:
            r = self.replay
            mps = self.replay_mps
            when = "death" if r.k > 0 else f"{r.seconds_left(mps):.1f}s before the death"
            y = self.text(f"Replay at {mps:g} moves/s: {when}" + ("" if r.playing else "  (paused)"),
                          x, y, colour=WARN)
        else:
            y = self.text("PAUSED" if paused else f"Speed {speed_label}  ({tps:.0f}/s)", x, y,
                          colour=ACCENT if paused else TEXT)
        y += 8

        for i, b in enumerate(a.boards):
            done = len(b.round_avgs)
            pygame.draw.rect(self.screen, MODEL_COLOURS[i], (x, y + 5, 10, 10))
            extra = (" + look-ahead" if b.agent.lookahead else "") + (" + route safety" if b.agent.shield else "")
            y = self.text(f"{b.name}  ({b.agent.senses}{extra})", x + 16, y, self.bold)
            if done:
                mean_avg = sum(b.round_avgs) / done
                mean_acc = sum(b.round_accs) / done
                y = self.text(f"avg {mean_avg:.1f}   best {b.best}   median {b.median:.0f}   "
                              f"acc {mean_acc * 100:.1f}%", x + 16, y, self.small)
                wins = f"{b.round_wins:g}" if len(a.boards) > 1 else "-"
                seeds = f"{100 * b.seed_wins / b.seeds_played:.0f}%" if len(a.boards) > 1 else "-"
                y = self.text(f"rounds won {wins} of {done}   sole best on {seeds} of seeds",
                              x + 16, y, self.small, DIM)
                speed = f", avg {b.fill_speed:,.0f} moves each" if b.fill_moves else ""
                y = self.text(f"boards filled: {b.boards_filled}{speed}", x + 16, y, self.small, DIM)
            else:
                y = self.text("results after the first round", x + 16, y, self.small, DIM)
            y += 6

        y += 4
        y = self.text("Average score per round", x, y, colour=ACCENT)
        y = self.chart(x, y + 4, w, 90) + 10

        y = self.text("Snake colour = move quality vs best possible move", x, y, self.small, DIM)
        grad = quality_colours(np.linspace(0, 1, w) ** (1 / C.COLOUR_POWER))[0]
        for i, col in enumerate(grad):
            pygame.draw.line(self.screen, col, (x + i, y + 2), (x + i, y + 14))
        y += 18
        self.text("worst", x, y, self.small, DIM)
        surf = self.small.render("best", True, DIM)
        self.screen.blit(surf, (x + w - surf.get_width(), y))
        y += 24
        if self.replay:
            keys = ("Space  play / pause", "Left / Right  one move back / forward",
                    "Up / Down  replay speed", "Home / End  10 s before / the death",
                    "R or Esc  leave replay")
        else:
            keys = ["Space  pause", "Up / Down  speed", "F  fastest",
                    "Tab  follow one snake (same apples on every board)", "Left / Right  other snake",
                    "R  replay followed snake's game"]
            if a.round_over:
                keys.append("N  next round")
            keys.append("Esc  back to menu")
        for k in keys:
            y = self.text(k, x, y, self.small, DIM)

    def chart(self, x, y, w, h):
        pygame.draw.rect(self.screen, CHECK_A, (x, y, w, h))
        series = [b.round_avgs[-w:] for b in self.arena.boards]
        if len(series[0]) >= 1:
            top = max(1.0, max(max(s) for s in series)) * 1.15
            for i, data in enumerate(series):
                if len(data) == 1:
                    pts = [(x, y + h - 2 - (data[0] / top) * (h - 6)), (x + w, y + h - 2 - (data[0] / top) * (h - 6))]
                else:
                    step = w / (len(data) - 1)
                    pts = [(x + j * step, y + h - 2 - (v / top) * (h - 6)) for j, v in enumerate(data)]
                pygame.draw.lines(self.screen, MODEL_COLOURS[i], False, pts, 2)
            self.text(f"{top:.0f}", x + 4, y + 2, self.small, DIM)
        else:
            self.text("fills in after round 1", x + 8, y + h // 2 - 9, self.small, DIM)
        return y + h


def run_compare(names, snakes, grid, seed=None):
    pygame.init()
    arena = Arena(names, snakes, grid, seed)
    window = Window(arena)
    speed = C.DEFAULT_SPEED_INDEX
    paused = False
    clock = pygame.time.Clock()
    owed = 0.0
    tps, tps_count, tps_time = 0.0, 0, time.perf_counter()
    running = True
    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                elif e.type != pygame.KEYDOWN:
                    continue
                elif e.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(speed + 1, len(C.SPEEDS) - 1)
                elif e.key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(speed - 1, 0)
                elif window.replay:                                # ---- replay controls
                    r = window.replay
                    if e.key in (pygame.K_ESCAPE, pygame.K_r):
                        window.replay = None
                        window.notice = ""
                    elif e.key == pygame.K_SPACE:
                        if r.k == r.end:
                            r.restart()
                        r.playing = not r.playing
                    elif e.key == pygame.K_RIGHT:
                        r.playing = False
                        r.step(1)
                    elif e.key == pygame.K_LEFT:
                        r.playing = False
                        r.step(-1)
                    elif e.key == pygame.K_HOME:
                        r.playing = False
                        r.restart()
                    elif e.key == pygame.K_END:
                        r.playing, r.k = False, r.end
                else:                                              # ---- live controls
                    if e.key == pygame.K_ESCAPE:
                        running = False
                    elif e.key == pygame.K_SPACE:
                        if arena.round_over:
                            arena.new_round()
                            window.focus_death_seen = False
                            paused = False
                        else:
                            paused = not paused
                        window.notice = ""
                    elif e.key == pygame.K_n and arena.round_over:
                        arena.new_round()
                        window.focus_death_seen = False
                        window.notice = ""
                        paused = False
                    elif e.key == pygame.K_f:
                        speed = len(C.SPEEDS) - 1 if C.SPEEDS[speed] else C.DEFAULT_SPEED_INDEX
                    elif e.key == pygame.K_TAB:
                        window.follow(0)
                        window.notice = ""
                    elif e.key == pygame.K_RIGHT:
                        window.follow(1)
                        window.notice = ""
                    elif e.key == pygame.K_LEFT:
                        window.follow(-1)
                        window.notice = ""
                    elif e.key == pygame.K_r:
                        window.replay_mps = C.SPEEDS[speed] or REPLAY_SPEED
                        window.start_replay(window.replay_mps)
                        window.notice = ""
                        paused = True

            target = C.SPEEDS[speed]
            if window.replay:
                r = window.replay
                window.replay_mps = target or REPLAY_SPEED
                if r.playing:
                    r.owed += window.replay_mps / C.FPS
                    if r.owed >= 1:
                        r.step(int(r.owed))
                        r.owed -= int(r.owed)
            elif not paused and not arena.round_over:
                if target == 0:                       # as fast as possible, redraw ~20x a second
                    deadline = time.perf_counter() + 0.05
                    while time.perf_counter() < deadline and not arena.round_over:
                        arena.tick()
                        tps_count += 1
                        if window.focus_dead_everywhere():
                            break
                else:
                    owed += target / C.FPS
                    while owed >= 1 and not arena.round_over:
                        owed -= 1
                        arena.tick()
                        tps_count += 1
                        if window.focus_dead_everywhere():
                            break
                # Stop and wait when the followed snake has died on every board.
                if window.focus_dead_everywhere() and not window.focus_death_seen:
                    window.focus_death_seen = True
                    paused = True
                    window.notice = (f"Snake #{window.focus + 1} has died on every board.   "
                                     f"R: replay its death   Space: carry on   Tab: stop following")
                if arena.round_over:
                    window.notice = "Round finished.   N or Space: next round   R: replay a snake's game"
            now = time.perf_counter()
            if now - tps_time >= 0.5:
                tps, tps_count, tps_time = tps_count / (now - tps_time), 0, now

            window.draw("fastest" if target == 0 else f"{target} ticks/s", tps, paused)
            clock.tick(C.FPS if (target or window.replay) else 0)
    finally:
        pygame.quit()
    return arena
