"""Console output for training and comparisons (used by menu.py, train.py and compare.py)."""

import os
import time

from . import config as C
from . import models


def _clock(secs):
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def train(name, snakes=None, grid=None, rounds=C.DEFAULT_ROUNDS, resume=False, seed=None,
          patience=C.PATIENCE_ROUNDS, senses=None, hidden=None, lookahead=None, curriculum=None,
          shield=None):
    from .trainer import Trainer                      # torch/numba import takes a moment
    print("\nLoading...", end="\r", flush=True)
    t = Trainer(name, snakes, grid, resume=resume, seed=seed, senses=senses, hidden=hidden,
                lookahead=lookahead, curriculum=curriculum, shield=shield)
    verb = "Continuing" if resume else "Training"
    print(f"{verb} '{name}' ({t.senses} senses, look-ahead {'on' if t.agent.lookahead else 'off'}"
          f"{', route safety on' if t.agent.shield else ''}): "
          f"{t.n} snakes on a {t.grid}x{t.grid} board, seed {t.seed}")
    stages = ", ".join(f"{n} ({lo}-{'full' if hi is None else hi})" for n, lo, hi in t.plan)
    print(f"Stages: {stages}. Up to {rounds} rounds each"
          + (f", moving on once a stage hasn't improved for {patience} rounds." if patience else "."))
    print("Every round gives every snake a brand new random seed (new apple positions).")
    used, total = t.threads
    print(f"Using {used} of {total} CPU threads (CPU_THREADS in config.py). "
          "Nothing is drawn, so it runs flat out. Ctrl+C stops early (progress is saved).")
    start = time.perf_counter()
    width = 0

    def live(rnd, tick, alive):
        nonlocal width
        line = f"  round {rnd:3d}   tick {tick:,}   playing {alive}/{t.n}"
        width = max(width, len(line))
        print(line.ljust(width), end="\r", flush=True)

    def report(r):
        top = "full" if r["hi"] is None else r["hi"]
        whole = "" if r["full_avg"] is None else f"   whole games avg {r['full_avg']:6.1f}"
        if r["fill_rate"]:
            whole += f"   filled {r['fill_rate']:.0%} in {r['fill_moves']:,.0f} moves"
        line = (f"  [{r['stage']} {r['lo']}-{top}] round {r['round']:3d}   gained {r['gain']:6.1f}{whole}"
                f"   best {r['best']:3d}   accuracy {r['accuracy'] * 100:5.1f}%   "
                f"random {r['epsilon'] * 100:4.1f}%   {r['seconds']:5.1f}s"
                + ("   new best" if r["new_best"] else ""))
        print(line.ljust(width))

    def status(text):
        print(("\n" + text).ljust(width))

    try:
        t.run(rounds, report=report, live=live, patience=patience, status=status)
    except KeyboardInterrupt:
        print("\n  Stopped early.")
    info = models.read_info(name)
    print(f"\nDone in {_clock(time.perf_counter() - start)}. '{name}' has now trained "
          f"{info.get('rounds', 0)} rounds, best whole-game average {info.get('best_avg', 0):.2f}.")
    print(f"Saved in {os.path.relpath(models.folder(name))}\n")


def compare(names, snakes=None, grid=None, seed=None):
    from .render import run_compare
    snakes = snakes or C.DEFAULT_COMPARE_SNAKES
    grid = grid or models.read_info(names[0]).get("grid") or C.DEFAULT_GRID
    print(f"\nComparing {', '.join(names)}: {snakes} snakes each on {grid}x{grid}. "
          f"Close the window or press Esc to come back here.")
    arena = run_compare(names, snakes, grid, seed)
    done = len(arena.boards[0].round_avgs)
    if not done:
        print("No round finished, so no results.\n")
        return
    games = len(arena.boards[0].scores)
    print(f"\nResults over {done} finished round(s) ({games} games per model), identical apples for every model:")
    print(f"  {'model':<22}{'avg':>8}{'best':>7}{'median':>8}{'filled':>8}{'moves/fill':>11}{'accuracy':>10}"
          f"{'rounds won':>12}{'sole best on':>14}")
    for b in arena.boards:
        seeds = f"{100 * b.seed_wins / b.seeds_played:.0f}% seeds" if len(arena.boards) > 1 else "-"
        speed = f"{b.fill_speed:,.0f}" if b.fill_moves else "-"
        print(f"  {b.name:<22}{sum(b.round_avgs) / done:>8.1f}{b.best:>7d}{b.median:>8.0f}{b.boards_filled:>8d}"
              f"{speed:>11}{sum(b.round_accs) / done * 100:>9.1f}%{b.round_wins:>12g}{seeds:>14}")
    print()


def solo(name, grid=None):
    from .solo import run_solo
    grid = grid or models.read_info(name).get("grid") or C.DEFAULT_GRID
    print(f"\nWatching {name} play one snake at a time on {grid}x{grid}, as fast as possible. "
          f"Close the window or press Esc to come back here.")
    s = run_solo(name, grid)
    n = len(s.scores)
    if not n:
        print("No game finished.\n")
        return
    filled = len(s.fill_moves)
    print(f"\n{name}: {n} games   average {sum(s.scores) / n:.1f}   best {max(s.scores)}   "
          f"boards filled {filled} ({filled / n:.0%})"
          + (f"   average {sum(s.fill_moves) / filled:,.0f} moves per full board" if filled else ""))
    print()

