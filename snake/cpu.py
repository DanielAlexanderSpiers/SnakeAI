"""How many CPU threads the path-finding (numba) and the brain (torch) may use."""

import os

import numba
import torch

from . import config as C


def limit_threads():
    """Apply CPU_THREADS: 0 = every thread, -2 = all but two (keeps the PC responsive),
    a positive number = exactly that many."""
    total = os.cpu_count() or 1
    want = C.CPU_THREADS if C.CPU_THREADS > 0 else total + C.CPU_THREADS
    want = max(1, min(want, total, numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(want)
    torch.set_num_threads(want)
    return want, total
