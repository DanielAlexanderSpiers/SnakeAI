import csv
import os
import secrets
import time

import numpy as np

from . import config as C
from . import models
from .agent import Agent, NStep
from .cpu import limit_threads
from .game import Snakes, DIED_WALL, DIED_SELF, DIED_STARVED, DIED_WON, FINISHED
from .lookahead import choose_moves
from .oracle import grade_moves
from .senses import observe, n_inputs, fenced_off, DEFAULT_SENSES, ROUTE_ORDER

LOG_HEADER = ["round", "stage", "ticks", "transitions", "epsilon", "stage_gain", "full_avg",
              "best_score", "accuracy", "died_wall", "died_self", "starved", "finished",
              "filled", "fill_moves", "loss", "seconds"]
NO_CAP = 10**9


def stage_plan(grid, curriculum):
    if not curriculum:
        return [("full", 0, None)]
    capacity = grid * grid - C.START_LENGTH
    bounds = [int(b) for b in str(C.STAGES).replace("/", ",").split(",") if b.strip()]
    edges = [0] + [b for b in bounds if 0 < b < capacity]
    names = {1: ["full"], 2: ["short", "long"], 3: ["short", "mid", "long"]}.get(
        len(edges), [f"stage{k + 1}" for k in range(len(edges))])
    return [(names[k], lo, edges[k + 1] if k + 1 < len(edges) else None) for k, lo in enumerate(edges)]


class Pool:

    def __init__(self, cap):
        self.cap, self.n, self.seen, self.data = cap, 0, 0, None

    def add(self, snap, rng):
        count = len(snap["length"])
        if count and self.data is None:
            self.data = {k: np.zeros((self.cap,) + v.shape[1:], dtype=v.dtype) for k, v in snap.items()}
        for j in range(count):
            self.seen += 1
            if self.n < self.cap:
                slot = self.n
                self.n += 1
            else:
                slot = int(rng.integers(self.seen))
                if slot >= self.cap:
                    continue
            for k, v in snap.items():
                self.data[k][slot] = v[j]

    def view(self):
        return {k: v[:self.n] for k, v in self.data.items()}

    def to_arrays(self, prefix):
        if not self.n:
            return {}
        out = {f"{prefix}{k}": v for k, v in self.view().items()}
        out[f"{prefix}seen"] = np.array([self.seen])
        return out

    def from_arrays(self, arrays, prefix):
        keys = [k[len(prefix):] for k in arrays if k.startswith(prefix) and k != f"{prefix}seen"]
        if not keys:
            return
        snap = {k: arrays[prefix + k][:self.cap] for k in keys}
        self.data = {k: np.zeros((self.cap,) + v.shape[1:], dtype=v.dtype) for k, v in snap.items()}
        self.n = len(snap["length"])
        for k, v in snap.items():
            self.data[k][:self.n] = v
        self.seen = int(arrays[f"{prefix}seen"][0])


class Trainer:
    def __init__(self, name, snakes=None, grid=None, resume=False, seed=None, senses=None, hidden=None,
                 lookahead=None, curriculum=None, shield=None):
        self.name = name
        self.threads = limit_threads()
        self.seed = int(seed) if seed is not None else secrets.randbits(62)
        info = models.read_info(name) if resume else {}
        self.n = snakes or info.get("snakes") or C.DEFAULT_SNAKES
        self.grid = grid or info.get("grid") or C.DEFAULT_GRID

        master = np.random.default_rng(self.seed)
        self.snakes = Snakes(self.n, self.grid, master)
        agent_rng = np.random.default_rng(master.integers(2**62))
        self.sample_rng = np.random.default_rng(master.integers(2**62))
        self.rng = np.random.default_rng(master.integers(2**62))
        self.dir = models.folder(name)
        os.makedirs(self.dir, exist_ok=True)
        if resume:
            self.agent = Agent.from_file(os.path.join(self.dir, "latest.pt"), agent_rng, self.seed,
                                         for_training=True)
            if lookahead is not None:
                self.agent.lookahead = lookahead
            if shield is not None:
                self.agent.shield = shield and self.agent.senses == "vision5"
        else:
            self.agent = Agent(agent_rng, self.seed, senses or DEFAULT_SENSES, hidden or C.HIDDEN,
                               lookahead=C.LOOKAHEAD if lookahead is None else lookahead,
                               shield=C.ROUTE_SHIELD if shield is None else shield)
        self.senses = self.agent.senses
        self.nstep = NStep(self.n, n_inputs(self.senses), C.N_STEP, C.GAMMA)

        if curriculum is None:
            curriculum = info.get("curriculum", C.CURRICULUM)
        self.plan = stage_plan(self.grid, curriculum)
        self.stage = min(info.get("stage", 0), len(self.plan) - 1) if resume else 0
        self.stage_start = self.agent.transitions
        cap = max(100, min(C.POOL_SIZE, int(50e6 // (self.grid * self.grid * 4))))
        self.pools = [Pool(cap) for _ in self.plan]
        pool_file = os.path.join(self.dir, "positions.npz")
        if resume and os.path.exists(pool_file) and info.get("grid") == self.grid:
            with np.load(pool_file) as arrays:
                for k, pool in enumerate(self.pools):
                    pool.from_arrays(arrays, f"s{k}_")

        self.info = info or dict(name=name, created=time.strftime("%Y-%m-%d %H:%M"),
                                 rounds=0, best_avg=0.0, best_score=0, seeds=[])
        self.info.update(snakes=self.n, grid=self.grid, senses=self.senses,
                         inputs=n_inputs(self.senses), hidden=self.agent.hidden,
                         lookahead=self.agent.lookahead, shield=self.agent.shield, curriculum=curriculum,
                         stages=[list(p) for p in self.plan])
        self.info["settings"] = {k: getattr(C, k) for k in (
            "GAMMA", "LEARNING_RATE", "BATCH_SIZE", "TRANSITIONS_PER_UPDATE", "EPS_END",
            "EPS_DECAY_TRANSITIONS", "SAFE_EXPLORATION", "REWARD_APPLE", "REWARD_WIN", "REWARD_DEATH",
            "REWARD_STARVE", "REWARD_CLOSER", "REWARD_FURTHER", "SHAPING_FADES", "STARVE_SLACK",
            "STARVE_SLACK_PER_SEGMENT", "STARVE_AREA_FACTOR", "N_STEP", "COMPACT_WEIGHT",
            "STAGES", "CONTROL_SHARE", "STAGE_EPS_START", "STAGE_EPS_DECAY", "MASTERED", "EARLIER_SHARE",
            "CYCLE_BUFFER", "ROUTE_WEIGHT")}
        self.info.setdefault("seeds", []).append(self.seed)
        self.round_no = self.info.get("rounds", 0)

        log_path = os.path.join(self.dir, "log.csv")
        if os.path.exists(log_path):
            with open(log_path) as f:
                header = f.readline().strip().split(",")
            if header != LOG_HEADER:
                os.replace(log_path, os.path.join(self.dir, "log_before_stages.csv"))
        new_log = not os.path.exists(log_path)
        self._log_file = open(log_path, "a", newline="")
        self._log = csv.writer(self._log_file)
        if new_log:
            self._log.writerow(LOG_HEADER)


    def run(self, rounds, report=print, live=None, patience=C.PATIENCE_ROUNDS, status=print):
        self.stopped_early = []
        try:
            for k in range(self.stage, len(self.plan)):
                self._enter_stage(k, status, live)
                marks, judged_on, best_smooth, stale = [], None, None, 0
                for _ in range(rounds):
                    r = self._play_round(live)
                    report(r)
                    now_on = "speed" if r["fill_rate"] >= 0.9 else "apples"
                    if now_on != judged_on:
                        if judged_on is not None:
                            status(f"  Nearly every game fills the board now: judging progress on speed.")
                        marks, judged_on, best_smooth, stale = [], now_on, None, 0
                    marks.append(r["gain"] if now_on == "apples" else 1e6 / max(1.0, r["fill_moves"]))
                    if not patience or self.agent.exploring or len(marks) < 5:
                        continue
                    smooth = sum(marks[-5:]) / 5
                    lo, hi = self.plan[k][1], self.plan[k][2]
                    if now_on == "apples" and hi is not None and smooth >= C.MASTERED * (hi - lo):
                        status(f"  Mastered: averaging {smooth:.1f} of the {hi - lo} apples this stage allows.")
                        break
                    if best_smooth is None or smooth > best_smooth * (1 + C.MIN_IMPROVEMENT):
                        best_smooth, stale = smooth, 0
                    else:
                        stale += 1
                        if stale >= patience:
                            self.stopped_early.append(self.plan[k][0])
                            break
                name = self.plan[k][0]
                self._save(f"stage{k + 1}-{name}.pt")
                status(f"Stage {k + 1} ({name}) done. Saved stage{k + 1}-{name}.pt")
                self._save_all()
            self.info["curriculum_done"] = True
        finally:
            self._save_all()
            self._log_file.close()

    def _enter_stage(self, k, status, live):
        self.stage = k
        self.info["stage"] = k
        name, lo, hi = self.plan[k]
        top = "the board is full" if hi is None else f"{hi} apples"
        status(f"Stage {k + 1} of {len(self.plan)}: {name} games ({lo} apples to {top})")
        if k > 0 and self.pools[k].n < C.POOL_MIN:
            self._top_up_pool(k, status, live)
        if k > 0 and self.pools[k].n:
            status(f"  starting from {self.pools[k].n} saved positions with {lo}+ apples, "
                   f"{C.CONTROL_SHARE:.0%} of snakes still play whole games")
        self.stage_start = self.agent.transitions

    @property
    def epsilon(self):
        if self.stage == 0 or self.agent.exploring:
            return self.agent.epsilon
        frac = min(1.0, (self.agent.transitions - self.stage_start) / C.STAGE_EPS_DECAY)
        return C.STAGE_EPS_START + frac * (C.EPS_END - C.STAGE_EPS_START)


    def _play_round(self, live):
        s, agent = self.snakes, self.agent
        k = self.stage
        name, lo, hi = self.plan[k]
        start = time.perf_counter()
        next_live = start + 0.25
        self.round_no += 1

        s.reset()
        cap = np.full(s.n, NO_CAP if hi is None else hi, dtype=np.int64)
        control = 0
        if k > 0:
            control = int(round(C.CONTROL_SHARE * s.n))
            cap[:control] = NO_CAP
            tier = np.arange(control, s.n)
            earlier = self.pools[k - 1] if k > 1 else None
            if earlier is not None and earlier.n and self.pools[k].n:
                n_early = int(round(C.EARLIER_SHARE * len(tier)))
                groups = [(tier[:n_early], earlier), (tier[n_early:], self.pools[k])]
            else:
                groups = [(tier, self.pools[k])]
            for who, pool in groups:
                if pool.n and len(who):
                    s.restore(who, pool.view(), self.rng.integers(0, pool.n, size=len(who)),
                              self.rng.integers(0, 2**62, size=len(who)))
        start_score = s.score.copy()
        later = np.array([p[1] for p in self.plan[1:]], dtype=np.int64)
        passed = start_score[:, None] >= later[None, :]

        obs = observe(s, self.senses)
        self.nstep.reset()
        tidy = -fenced_off(s)
        use_route = self.senses == "vision5" and C.ROUTE_WEIGHT
        order = obs[:, ROUTE_ORDER].copy() if use_route else None
        ticks = 0
        grade_sum, grade_count = 0.0, 0

        while s.alive.any():
            alive = np.flatnonzero(s.alive)
            states = obs[alive]
            safe = s.safe_moves()[alive] if C.SAFE_EXPLORATION else None
            acts = choose_moves(agent, s, alive, states, self.epsilon, self.rng, safe)
            actions = np.zeros(s.n, dtype=np.int64)
            actions[alive] = acts

            pick = alive if len(alive) <= C.GRADE_SAMPLE else \
                self.sample_rng.choice(alive, C.GRADE_SAMPLE, replace=False)
            which = np.zeros(s.n, dtype=bool)
            which[pick] = True
            grade_sum += float(grade_moves(s, which)[pick, actions[pick]].sum())
            grade_count += len(pick)

            reward, done, _ = s.step(actions)
            obs = observe(s, self.senses)
            rew, dead = reward[alive], done[alive]
            if C.COMPACT_WEIGHT:
                new_tidy = -fenced_off(s)[alive]
                new_tidy[dead] = 0.0
                rew = rew + C.COMPACT_WEIGHT * (C.GAMMA * new_tidy - tidy[alive])
                tidy[alive] = new_tidy
            if use_route:
                new_order = obs[alive, ROUTE_ORDER].copy()
                new_order[dead] = 0.0
                rew = rew + C.ROUTE_WEIGHT * (C.GAMMA * new_order - order[alive])
                order[alive] = new_order
            ready = self.nstep.push(alive, states, acts, rew, obs[alive], dead)
            if ready is not None:
                agent.remember(*ready)
            self._keep_positions(passed, later)
            done_here = np.flatnonzero(s.alive & (s.score >= cap))
            if len(done_here):
                s.finish(done_here)
                self.nstep.drop(done_here)
            agent.learn()
            ticks += 1

            if live and time.perf_counter() > next_live:
                live(self.round_no, ticks, len(alive))
                next_live = time.perf_counter() + 0.25

        tier = np.arange(control, s.n)
        gain = float((s.score[tier] - start_score[tier]).mean())
        full_avg = float(s.score[:control].mean()) if control else (gain if hi is None else None)
        filled = s.death[tier] == DIED_WON
        fill_rate = float(filled.mean())
        fill_moves = float(s.t[tier][filled].mean()) if filled.any() else 0.0
        whole = np.arange(control) if control else tier
        full_moves = float(s.t[whole].mean())
        best = int(s.score.max())
        acc = grade_sum / max(1, grade_count)
        deaths = [int((s.death == c).sum()) for c in (DIED_WALL, DIED_SELF, DIED_STARVED, FINISHED)]
        secs = time.perf_counter() - start
        self._log.writerow([self.round_no, name, ticks, agent.transitions, f"{self.epsilon:.4f}",
                            f"{gain:.3f}", "" if full_avg is None else f"{full_avg:.3f}", best,
                            f"{acc:.4f}", *deaths, int(filled.sum()), f"{fill_moves:.0f}",
                            f"{agent.last_loss:.5f}", f"{secs:.2f}"])
        self._log_file.flush()

        self.info["rounds"] = self.round_no
        self.info["transitions"] = agent.transitions
        self.info["best_score"] = max(self.info.get("best_score", 0), best)
        old = self.info.get("best_avg", 0)
        improved = (full_avg is not None and agent.transitions > C.WARMUP_TRANSITIONS and
                    (full_avg > old + 0.05 or
                     (full_avg >= old - 0.05 and full_moves < self.info.get("best_moves", float("inf")))))
        if improved:
            self.info["best_avg"] = max(full_avg, old)
            self.info["best_moves"] = full_moves
            self._save("best.pt")
        if self.round_no % C.SAVE_EVERY_ROUNDS == 0:
            self._save_all()

        return dict(round=self.round_no, stage=name, stage_no=k + 1, stages=len(self.plan), lo=lo, hi=hi,
                    gain=gain, full_avg=full_avg, best=best, accuracy=acc, epsilon=self.epsilon,
                    wall=deaths[0], self=deaths[1], starved=deaths[2], finished=deaths[3],
                    ticks=ticks, seconds=secs, new_best=improved, fill_rate=fill_rate,
                    fill_moves=fill_moves)

    def _keep_positions(self, passed, later):
        s = self.snakes
        for j, at in enumerate(later):
            new = np.flatnonzero(s.alive & ~passed[:, j] & (s.score >= at))
            if len(new):
                passed[new, j] = True
                self.pools[j + 1].add(s.snapshot(new), self.rng)

    def _top_up_pool(self, k, status, live):
        lo = self.plan[k][1]
        s, agent = self.snakes, self.agent
        status(f"  collecting positions with {lo}+ apples to start from "
               f"(have {self.pools[k].n}, want {C.POOL_MIN})...")
        later = np.array([p[1] for p in self.plan[1:]], dtype=np.int64)
        for attempt in range(20):
            s.reset()
            prev = self.pools[k - 1] if k > 1 else None
            if prev is not None and prev.n:
                s.restore(np.arange(s.n), prev.view(), self.rng.integers(0, prev.n, size=s.n),
                          self.rng.integers(0, 2**62, size=s.n))
            passed = s.score[:, None] >= later[None, :]
            before, ticks = self.pools[k].n, 0
            while s.alive.any():
                alive = np.flatnonzero(s.alive)
                obs = observe(s, self.senses)
                acts = choose_moves(agent, s, alive, obs[alive], 0.0, self.rng)
                actions = np.zeros(s.n, dtype=np.int64)
                actions[alive] = acts
                s.step(actions)
                self._keep_positions(passed, later)
                s.finish(np.flatnonzero(s.alive & (s.score >= lo)))
                ticks += 1
                if live and ticks % 200 == 0:
                    live(self.round_no, ticks, len(alive))
            status(f"  ...{self.pools[k].n} positions")
            if self.pools[k].n >= C.POOL_MIN or self.pools[k].n == before:
                break
        if not self.pools[k].n:
            status(f"  the model never reached {lo} apples, so this stage starts from fresh games")


    def _save(self, filename):
        self.agent.save(os.path.join(self.dir, filename),
                        extra=dict(name=self.name, round=self.round_no, grid=self.grid, snakes=self.n,
                                   stage=self.stage))

    def _save_all(self):
        self._save("latest.pt")
        arrays = {}
        for k, pool in enumerate(self.pools):
            arrays.update(pool.to_arrays(f"s{k}_"))
        if arrays:
            np.savez_compressed(os.path.join(self.dir, "positions.npz"), **arrays)
        models.write_info(self.name, self.info)
