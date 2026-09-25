import argparse

from snake import config as C
from snake import console, models

p = argparse.ArgumentParser(description="Watch 1-4 models play the exact same games")
p.add_argument("names", nargs="*", help="model names (default: the newest model)")
p.add_argument("--snakes", type=int, help=f"snakes per model (default {C.DEFAULT_COMPARE_SNAKES})")
p.add_argument("--grid", type=int, help="board size (default: the first model's training grid)")
p.add_argument("--seed", type=int, help="replay specific games (default: random)")
a = p.parse_args()

names = a.names or [n for n, _ in models.list_models()[:1]]
if not names:
    raise SystemExit("No trained models yet. Run train.py first.")
if len(names) > 4:
    raise SystemExit("Up to 4 models at once.")
for n in names:
    if not models.exists(n):
        raise SystemExit(f"No model called '{n}'.")
console.compare(names, a.snakes, a.grid, a.seed)
