"""Named models on disk. Each lives in models/<name>/ with:

    best.pt      the brain from its best round (highest average score)
    latest.pt    the brain at the end of its most recent training
    log.csv      one row per training round
    info.json    settings and headline numbers
"""

import json
import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
NAME_OK = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def folder(name):
    return os.path.join(MODELS, name)


def exists(name):
    return os.path.exists(os.path.join(folder(name), "latest.pt"))


def default_name():
    return time.strftime("model-%Y%m%d-%H%M")


def read_info(name):
    try:
        with open(os.path.join(folder(name), "info.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_info(name, info):
    os.makedirs(folder(name), exist_ok=True)
    with open(os.path.join(folder(name), "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def list_models():
    """[(name, info)] for every trained model, newest first."""
    if not os.path.isdir(MODELS):
        return []
    names = [n for n in os.listdir(MODELS) if exists(n)]
    names.sort(key=lambda n: os.path.getmtime(os.path.join(folder(n), "latest.pt")), reverse=True)
    return [(n, read_info(n)) for n in names]


def brain_path(name):
    """best.pt if the model has one, otherwise latest.pt."""
    best = os.path.join(folder(name), "best.pt")
    return best if os.path.exists(best) else os.path.join(folder(name), "latest.pt")


def describe(info):
    if not info:
        return ""
    senses = info.get("senses") or ("basic" if info.get("inputs", 11) == 11 else "?")
    if info.get("lookahead"):
        senses += "+look"
    if info.get("shield"):
        senses += "+safe"
    return (f"[{senses}] {info.get('rounds', 0)} rounds, best avg {info.get('best_avg', 0):.1f}, "
            f"best {info.get('best_score', 0)}, trained on {info.get('snakes', '?')} snakes "
            f"{info.get('grid', '?')}x{info.get('grid', '?')}")
