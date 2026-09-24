"""Deep Q-Learning. Every snake on the board shares this one brain and one memory.

Each tick, every living snake adds one experience (what it saw, what it did, the
reward, what it saw next) to the replay memory. For every TRANSITIONS_PER_UPDATE new
experiences the brain takes one training step on a random batch from that memory.
Tying training to experience (not ticks) keeps the cost honest: a tick with 500
snakes alive trains about once, a tick with 3 stragglers left barely trains at all.

Double DQN with a slowly-following target network keeps the Q-value targets stable.
"""

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C
from .brain import Brain, N_ACTIONS
from .senses import n_inputs, SENSES


class Replay:
    def __init__(self, capacity, width):
        self.cap = capacity
        self.s = np.zeros((capacity, width), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, width), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    def __len__(self):
        return self.cap if self.full else self.pos

    def add(self, s, a, r, s2, done):
        k = len(a)
        at = (self.pos + np.arange(k)) % self.cap
        self.s[at], self.a[at], self.r[at], self.s2[at], self.done[at] = s, a, r, s2, done
        if self.pos + k >= self.cap:
            self.full = True
        self.pos = (self.pos + k) % self.cap

    def sample(self, k, rng):
        i = rng.integers(0, len(self), size=k)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.done[i]


class NStep:
    """Turns one-move experiences into N-move ones, per snake: the rewards of the next N
    moves added up (discounted), then the brain's own estimate from where it ended up.
    When a snake dies the moves still waiting are flushed with what they got."""

    def __init__(self, n_snakes, width, n, gamma):
        self.n, self.gamma = n, gamma
        self.s = np.zeros((n_snakes, n, width), dtype=np.float32)
        self.a = np.zeros((n_snakes, n), dtype=np.int64)
        self.r = np.zeros((n_snakes, n), dtype=np.float32)
        self.count = np.zeros(n_snakes, dtype=np.int64)
        self.powers = (gamma ** np.arange(n)).astype(np.float32)

    def reset(self):
        self.count[:] = 0

    def drop(self, idx):
        """Forget moves still waiting for these snakes (their game was stopped, not lost)."""
        self.count[idx] = 0

    def push(self, idx, s, a, r, s2, done):
        """idx: which snakes moved; the rest are aligned with idx. Returns ready experiences
        (s, a, summed reward, s after N moves, done)."""
        out = []
        c = self.count[idx]
        self.s[idx, c] = s
        self.a[idx, c] = a
        self.r[idx, c] = r
        self.count[idx] = c + 1

        full = self.count[idx] == self.n
        if full.any():
            rows = idx[full]
            ret = (self.r[rows] * self.powers).sum(1)
            out.append((self.s[rows, 0].copy(), self.a[rows, 0].copy(), ret, s2[full],
                        done[full].astype(np.float32)))
            self.s[rows, :-1] = self.s[rows, 1:]
            self.a[rows, :-1] = self.a[rows, 1:]
            self.r[rows, :-1] = self.r[rows, 1:]
            self.count[rows] -= 1

        if done.any():
            rows = idx[done]
            steps = np.arange(self.n)
            for j in range(self.n):
                has = self.count[rows] > j
                if not has.any():
                    break
                rr = rows[has]
                live = (steps[None, :] >= j) & (steps[None, :] < self.count[rr][:, None])
                disc = self.gamma ** np.clip(steps - j, 0, None)
                ret = (self.r[rr] * live * disc[None, :]).sum(1).astype(np.float32)
                out.append((self.s[rr, j].copy(), self.a[rr, j].copy(), ret, self.s[rr, j].copy(),
                            np.ones(len(rr), dtype=np.float32)))
            self.count[rows] = 0

        if not out:
            return None
        return tuple(np.concatenate(parts) for parts in zip(*out))


def read_arch(path):
    """(senses, hidden, lookahead) of a saved brain. Older files don't record all of it,
    so fill the gaps from the weights."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    arch = data.get("arch")
    if arch:
        return arch["senses"], arch["hidden"], arch.get("lookahead", False), arch.get("shield", False)
    hidden, width = data["online"]["net.0.weight"].shape
    senses = next(k for k, v in SENSES.items() if v == width)
    return senses, int(hidden), False, False


class Agent:
    def __init__(self, rng, seed, senses="basic", hidden=C.HIDDEN, memory=True, lookahead=False,
                 shield=False):
        torch.manual_seed(seed)
        self.rng = rng
        self.senses, self.hidden = senses, hidden
        self.lookahead = lookahead            # try each move out before choosing (see lookahead.py)
        self.shield = shield and senses == "vision5"   # route safety (see lookahead.py)
        width = n_inputs(senses)
        self.online = Brain(width, hidden)
        self.target = Brain(width, hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=C.LEARNING_RATE)
        self.memory = Replay(C.REPLAY_CAPACITY if memory else 1, width)
        self.transitions = 0     # total experiences ever collected
        self.updates = 0
        self.owed_updates = 0.0
        self.last_loss = float("nan")

    @property
    def exploring(self):
        """Still in the random-exploration phase at the start of training."""
        return self.transitions < C.EPS_DECAY_TRANSITIONS

    @property
    def epsilon(self):
        if not self.exploring:
            return C.EPS_END
        frac = self.transitions / C.EPS_DECAY_TRANSITIONS
        return C.EPS_START + frac * (C.EPS_END - C.EPS_START)

    def q_values(self, states):
        """The brain's predicted future reward for each move: (n, 3)."""
        with torch.no_grad():
            return self.online(torch.from_numpy(np.ascontiguousarray(states))).numpy()

    def act(self, states, epsilon, safe=None):
        """Greedy move from the brain, replaced by a random move with probability epsilon.
        safe: optional (n, 3) bool; random moves then only pick non-fatal moves when possible."""
        return self.explore(self.q_values(states).argmax(1), epsilon, safe)

    def explore(self, actions, epsilon, safe=None):
        """Swap a random share (epsilon) of the chosen moves for random ones."""
        actions = np.array(actions, dtype=np.int64)
        if epsilon > 0:
            explore = np.flatnonzero(self.rng.random(len(actions)) < epsilon)
            if len(explore):
                weights = np.ones((len(explore), N_ACTIONS))
                if safe is not None:
                    ok = safe[explore]
                    weights = np.where(ok.any(1, keepdims=True), ok, 1.0)
                cum = np.cumsum(weights, 1)
                pick = self.rng.random(len(explore)) * cum[:, -1]
                actions[explore] = (pick[:, None] >= cum).sum(1)
        return actions

    def remember(self, s, a, r, s2, done):
        self.memory.add(s, a, r, s2, done)
        self.transitions += len(a)
        self.owed_updates += len(a) / C.TRANSITIONS_PER_UPDATE

    def learn(self):
        if len(self.memory) < max(C.WARMUP_TRANSITIONS, C.BATCH_SIZE):
            self.owed_updates = 0.0
            return
        while self.owed_updates >= 1:
            self.owed_updates -= 1
            s, a, r, s2, done = (torch.from_numpy(x) for x in self.memory.sample(C.BATCH_SIZE, self.rng))
            q = self.online(s).gather(1, a[:, None]).squeeze(1)
            with torch.no_grad():
                best_next = self.online(s2).argmax(1, keepdim=True)
                q_next = self.target(s2).gather(1, best_next).squeeze(1)
                target = r + (C.GAMMA ** C.N_STEP) * (1.0 - done) * q_next
            loss = F.smooth_l1_loss(q, target)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), C.GRAD_CLIP)
            self.opt.step()
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.lerp_(op, C.TARGET_TAU)
            self.updates += 1
            self.last_loss = loss.item()

    # ------------------------------------------------------------ saving

    def save(self, path, extra=None):
        torch.save({
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "opt": self.opt.state_dict(),
            "transitions": self.transitions,
            "updates": self.updates,
            "arch": {"senses": self.senses, "hidden": self.hidden, "lookahead": self.lookahead,
                     "shield": self.shield},
            "extra": extra or {},
        }, path)

    @classmethod
    def from_file(cls, path, rng=None, seed=0, for_training=False):
        """Build an agent with whatever senses/size the saved brain was trained with, and load it."""
        senses, hidden, lookahead, shield = read_arch(path)
        agent = cls(rng if rng is not None else np.random.default_rng(seed), seed,
                    senses, hidden, memory=for_training, lookahead=lookahead, shield=shield)
        agent.load(path, weights_only_brain=not for_training)
        return agent

    def load(self, path, weights_only_brain=False):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.online.load_state_dict(data["online"])
        self.target.load_state_dict(data.get("target", data["online"]))
        if not weights_only_brain:
            self.opt.load_state_dict(data["opt"])
            self.transitions = data.get("transitions", 0)
            self.updates = data.get("updates", 0)
        return data.get("extra", {})
