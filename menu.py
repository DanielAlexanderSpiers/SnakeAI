"""Interactive menu. run.bat starts this."""

import os
import shutil
import sys

from snake import config as C
from snake import console, models


def ask(prompt, default, cast=str, check=None, error="Try again."):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError:
            print(f"  {error}")
            continue
        if check is None or check(value):
            return value
        print(f"  {error}")


def yes_no(prompt, default):
    raw = input(f"{prompt}? [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not raw else raw.startswith("y")


def pick_model(listing, prompt="Model number"):
    num = ask(prompt, 1, int, lambda v: 1 <= v <= len(listing), f"Pick 1-{len(listing)}.")
    return listing[num - 1][0]


def show_models(listing):
    for i, (name, info) in enumerate(listing, 1):
        print(f"  {i}) {name:<24} {models.describe(info)}")


def train_new():
    name = ask("Model name", models.default_name(), str, models.NAME_OK.match,
               "Letters, numbers, - and _ only (max 40).")
    if models.exists(name):
        print(f"  '{name}' already exists. Use 'Continue training' for it, or pick another name.")
        return
    snakes = ask("Snakes at once", C.DEFAULT_SNAKES, int, lambda v: 1 <= v <= 5000, "1-5000.")
    grid = ask("Grid size", C.DEFAULT_GRID, int, lambda v: 10 <= v <= 200, "10-200.")
    print("  Senses:  1) vision5 - vision4, plus a route through every square (the way to fill the board)")
    print("           2) vision4 - vision3, plus tail distance, fenced-off apple, board split")
    print("           3) vision3 - vision2, plus tidiness: hugging walls, not fencing off space")
    print("           4) vision2 - vision, plus looks ahead: can it escape, can it eat safely")
    print("           5) vision  - sees how much room each move leaves and where its body will be")
    print("           6) basic   - the original 11 inputs, only sees the squares next to its head")
    pick = ask("Senses", 1, int, lambda v: 1 <= v <= 6, "1 to 6.")
    if pick == 1 and grid % 2:
        print(f"  Note: a {grid}x{grid} board has no route through every square (odd size), so the")
        print("  route senses will be blank. Use an even grid size to get the benefit.")
    lookahead = yes_no("Look-ahead: try every move out before choosing (~3x slower, stronger)", True)
    shield = False
    if pick == 1:
        print("  Route safety: when a move exists that keeps the body in route order, the brain may")
        print("  only choose among those. That alone guarantees filling the board; the brain then")
        print("  learns which safe moves fill it fastest. Say no to let it learn safety by itself.")
        shield = yes_no("Route safety", True)
    stages = yes_no("Train in stages: short (0-50 apples), mid (50-150), long (150-full)", True)
    print("  Each stage moves on by itself once it stops improving.")
    rounds = ask("Maximum rounds" + (" per stage" if stages else ""), C.DEFAULT_ROUNDS, int,
                 lambda v: v >= 1, "At least 1.")
    console.train(name, snakes, grid, rounds, senses=["vision5", "vision4", "vision3", "vision2", "vision", "basic"][pick - 1],
                  lookahead=lookahead, curriculum=stages, shield=shield)


def train_more(listing):
    show_models(listing)
    name = pick_model(listing)
    info = models.read_info(name)
    lookahead = yes_no("Look-ahead: try every move out before choosing (~3x slower, stronger)",
                       bool(info.get("lookahead", False)))
    if info.get("stage") is not None:
        print(f"  Carries on from stage {info['stage'] + 1} of its training.")
    rounds = ask("Maximum extra rounds per stage", 50, int, lambda v: v >= 1, "At least 1.")
    console.train(name, rounds=rounds, resume=True, lookahead=lookahead)


def compare(listing):
    show_models(listing)
    default = "1 2" if len(listing) >= 2 else "1"

    def parse(raw):
        nums = [int(p) for p in raw.replace(",", " ").split()]
        return nums

    def valid(nums):
        return 1 <= len(nums) <= 4 and len(set(nums)) == len(nums) and \
            all(1 <= n <= len(listing) for n in nums)

    nums = ask("Models to compare, e.g. 1 2 (up to 4)", default, parse, valid,
               f"Give 1-4 different numbers from 1-{len(listing)}.")
    if isinstance(nums, str):
        nums = parse(nums)
    names = [listing[n - 1][0] for n in nums]
    snakes = ask("Snakes per model", C.DEFAULT_COMPARE_SNAKES, int, lambda v: 1 <= v <= 2000, "1-2000.")
    grid = ask("Grid size", models.read_info(names[0]).get("grid", C.DEFAULT_GRID), int,
               lambda v: 10 <= v <= 200, "10-200.")
    console.compare(names, snakes, grid)


def watch_solo(listing):
    show_models(listing)
    name = pick_model(listing)
    grid = ask("Grid size", models.read_info(name).get("grid", C.DEFAULT_GRID), int,
               lambda v: 10 <= v <= 200, "10-200.")
    console.solo(name, grid)


def delete_models():
    """Pick which trained models to delete for good."""
    listing = models.list_models()
    if not listing:
        print("  There are no models to delete.")
        return
    show_models(listing)
    raw = input("Models to delete, e.g. 1 3, or 'all' (Enter to cancel): ").strip().lower()
    if not raw:
        print("  Nothing deleted.")
        return
    if raw == "all":
        picked = [name for name, _ in listing]
    else:
        try:
            nums = sorted({int(p) for p in raw.replace(",", " ").split()})
        except ValueError:
            print("  Those aren't model numbers. Nothing deleted.")
            return
        if not nums or not all(1 <= n <= len(listing) for n in nums):
            print(f"  Pick numbers from 1 to {len(listing)}. Nothing deleted.")
            return
        picked = [listing[n - 1][0] for n in nums]
    print("  About to delete for good: " + ", ".join(picked))
    if input("  Type yes to delete them: ").strip().lower() != "yes":
        print("  Nothing deleted.")
        return
    for name in picked:
        shutil.rmtree(models.folder(name), ignore_errors=True)
        print(f"  Deleted {name}" if not os.path.exists(models.folder(name))
              else f"  Could not delete {name} (is it open somewhere?)")


def main():
    while True:
        listing = models.list_models()
        print("=" * 60)
        print("  SnakeAI")
        print("=" * 60)
        if listing:
            print(f"  {len(listing)} trained model(s) in {os.path.relpath(models.MODELS)}")
        print("  1) Train a new model")
        print("  2) Continue training a model")
        print("  3) Compare models (watch them play)")
        print("  4) Watch one snake at full speed")
        print("  5) Delete models")
        print("  6) Quit")
        choice = input("> ").strip()
        try:
            if choice == "1":
                train_new()
            elif choice in ("2", "3", "4"):
                if not listing:
                    print("  No trained models yet. Train one first (option 1).")
                elif choice == "2":
                    train_more(listing)
                elif choice == "3":
                    compare(listing)
                else:
                    watch_solo(listing)
            elif choice == "5":
                delete_models()
            elif choice in ("6", "q", "quit", "exit"):
                return
        except KeyboardInterrupt:
            print("\n  Cancelled.")
        print()


if __name__ == "__main__":
    try:
        if "--delete" in sys.argv:
            delete_models()
        else:
            main()
    except (KeyboardInterrupt, EOFError):
        pass
