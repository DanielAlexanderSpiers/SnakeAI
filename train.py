"""Train a model from the command line. Or just use run.bat."""

import argparse

from snake import config as C
from snake import console, models

p = argparse.ArgumentParser(description="Train a snake model (no graphics, full speed)")
p.add_argument("--name", default=None, help="model name (default: model-<date>-<time>)")
p.add_argument("--snakes", type=int, help=f"snakes at once (default {C.DEFAULT_SNAKES})")
p.add_argument("--grid", type=int, help=f"board size (default {C.DEFAULT_GRID})")
p.add_argument("--rounds", type=int, default=C.DEFAULT_ROUNDS, help=f"maximum rounds (default {C.DEFAULT_ROUNDS})")
p.add_argument("--patience", type=int, default=C.PATIENCE_ROUNDS,
               help=f"stop after this many rounds without improvement, 0 = never (default {C.PATIENCE_ROUNDS})")
p.add_argument("--resume", action="store_true", help="carry on training an existing model")
p.add_argument("--seed", type=int, help="replay a specific run (default: random)")
p.add_argument("--senses", choices=["vision5", "vision4", "vision3", "vision2", "vision", "basic"], default=None,
               help="what the snake can see (default vision5; ignored with --resume)")
p.add_argument("--no-lookahead", action="store_true", help="don't try moves out before choosing")
p.add_argument("--no-stages", action="store_true", help="train on whole games instead of short/mid/long")
p.add_argument("--no-route-safety", action="store_true", help="vision5: let it learn route safety by itself")
p.add_argument("--hidden", type=int, default=None, help=f"brain size (default {C.HIDDEN})")
a = p.parse_args()

name = a.name or models.default_name()
if a.resume and not models.exists(name):
    raise SystemExit(f"No model called '{name}'.")
if not a.resume and models.exists(name):
    raise SystemExit(f"'{name}' already exists. Add --resume to keep training it.")
console.train(name, a.snakes, a.grid, a.rounds, resume=a.resume, seed=a.seed, patience=a.patience,
              senses=a.senses, hidden=a.hidden, lookahead=False if a.no_lookahead else None,
              curriculum=False if a.no_stages else None, shield=False if a.no_route_safety else None)
