# SnakeAI - plan (as built)

Hundreds of snakes, all driven by a single neural network that learns by
reinforcement learning (Deep Q-Learning). Training runs flat out with nothing drawn;
trained models are then watched side by side on identical games to compare them.

**The goal: fill every square of the board with the snake, as fast as possible.**
A snake that does is a win (+100 reward, shown as FILLED THE BOARD).

## How it works

- **One shared brain.** Every snake is controlled by the same network. Each move
  every living snake makes goes into one shared replay memory, and the network
  trains on random batches of that memory. 500 snakes = 500 times the experience
  per tick.
- **Rewards:** +100 filling the board, +10 apple, -10 wall or own body, -20 starving,
  +0.1 for a step towards the apple and -0.15 for a step away. The towards/away nudge
  fades out as the snake fills the board, because a long snake often has to go the
  long way round to stay safe.
- **Starvation.** Each apple gives a move budget of at least one move per square on
  the board (400 on 20x20), or `distance + 10 + length` if that's more. A snake always
  has time to go all the way round to stay safe; starvation only catches endless loops.
- **Exploration.** Starts at 100% random moves, falling to 0.1% over the first
  400,000 moves (about 10 rounds with 1000 snakes). Random moves never pick an
  instantly fatal move when a safe one exists (safe exploration), so long games
  aren't thrown away and the brain gets real late-game practice.
- **Double DQN** with a slowly-following target network, Huber loss, Adam,
  gamma 0.98 (looks ~50 moves ahead, enough to feel a trap coming).
- **Look-ahead.** Before every move (in training and when watching) each snake tries
  all three moves on copies of the board, and the brain judges the position each one
  leads to: reward + gamma x best Q from there. It sees a trap one move before walking
  into it. About 3.6x the work per move. The copies place new apples with their own
  random seeds, so it never learns where the real next apple will be.
- **Training in stages.** Short games (0-50 apples), then mid (50-150), then long
  (150 until the board is full), each until it stops improving. Mid and long games start
  from real positions the model reached itself (kept whenever a snake passes 50 / 150).
  20% of snakes keep playing whole games from the start so nothing is forgotten and
  best.pt is picked on real full-game scores. Each stage also saves stageN-name.pt.
  A stage also ends once it's mastered (5-round average within 3% of its top, e.g. 48.5
  of 50), and in the long stage 30% of games start from mid-length positions so the mid
  game isn't unlearned (v4's long stage slowly lost ~10 apples of full-game average).
- **3-step learning.** Each experience covers the next 3 moves, so a death is felt by
  the moves that set it up, not only the last one.
- **Tidiness reward.** Moves that fence part of the board off from the head cost a
  little; reopening it earns the same back (potential-based shaping, so the best way
  to play doesn't change; it teaches neat packing sooner). `COMPACT_WEIGHT`, 0 = off.

## Fairness: the only luck is apple placement

- Every snake moves exactly one cell per tick, so all snakes move at the same speed.
- Every snake spawns in the same place (centre, facing up, length 3).
- Every snake has its own random seed, re-rolled each round. Apple number k for a
  seed always comes from the random stream (seed, k), so the same seed offers the
  same apples in the same order **whatever the brain does**.
- Snakes never interact. Each has exactly one apple at a time (a red square).
- Wall or own body kills the snake and it vanishes, along with its apple.

## Seeds

- **Training:** every run starts from a fresh random master seed. Every round, every
  snake gets a brand new seed from it, so apple layouts are never repeated.
- **Comparing:** a fresh random seed every time the window opens and every round, but
  snake #N gets the *same* seed on every model's board, so they face identical games.

## Training vs watching

- **Training draws nothing** and runs flat out in the console. It **stops by itself**
  once the 5-round average score hasn't improved by 2% for 10 rounds (the 11-input
  brain plateaus after ~10-15 rounds). 1000 snakes on 20x20 now trains in ~30 seconds
  (was 5.5 minutes for the same result).
- Speed-ups: one training step per 1024 new experiences, apple placement compiled with
  numba, referee only grades a sample of 64 snakes per tick during training, and the
  vision senses do one combined flood fill per move (1.6x faster, identical numbers).
- **CPU, not GPU.** Training time for a vision2 model: 53% path-finding (numba, CPU),
  24% neural network (torch), the rest game rules. The network is tiny (57-128-128-3),
  so on the RTX 3060 Ti copying data to the GPU and back costs about what the maths
  saves: 10% faster at best, for a 2.5 GB CUDA install. Not worth it.
  `CPU_THREADS` (default 0 = every thread; -2 leaves two free for the rest of the PC).
- **Watching is for comparing models.** Pick 1-4 trained models; each gets its own
  board, and snake #N on every board plays the same seed. No random moves and no
  training, so any difference is down to the brain. Tab follows one snake (others
  fade out) and shows each board's verdict on its last move.
- **Watch one snake at full speed** (menu option 4): one model, one snake, no referee,
  redrawn ~30 times a second while the rest of the time plays moves. It uses its own
  engine (`fast.py`): the game step compiled for one snake, the brain as plain numpy maths,
  one reused set of look-ahead copies, and with route safety it skips deciding entirely
  when only one move is allowed. About 24,000 moves/s for V5 on 20x20, so a whole board
  fill takes ~1.5 seconds (the first version managed 3,300/s). Checked move for move
  against normal play over whole games. Filling the board stops on an end screen
  (moves, seconds of play, moves per apple) until Space / N. Games follow each
  other automatically; the panel keeps average, best, median, boards filled and moves per
  full board. It runs on one CPU thread (faster than 12 for a single snake) and, with
  look-ahead, reuses the copy that already tried the chosen move instead of playing it a
  second time (checked move-for-move identical over a whole 36,117-move game).
- **Nothing restarts by itself.** A finished round waits for N. Following a snake
  pauses when it has died on every board.
- **Replays.** R replays the last 10 seconds of the followed snake's life on every
  board at once, lined up so both deaths happen at the same moment (at the current
  speed, e.g. 200 moves at 20 moves/s). Left/Right step one move (and further back
  than 10 s if you want), Home/End jump, the fatal move is marked with an X and the
  label says how it died and what the referee thought of every move on the way.

## Colour = move quality

A referee (`oracle.py`) works out the **best possible move** every tick using the
safe-greedy strategy:

1. Take the shortest path to the apple **only if it is safe**: replay the path on a
   copy of the board, grow by one, and check the snake can still escape afterwards.
2. Otherwise make any move the snake can escape from (stay alive, wait for a way).
3. If nothing is escapable the snake is already doomed by an earlier move.

**Escaping** is time-aware (`paths.py`): being shut in by your own body is fine if the
space you're in is big enough to keep moving until that body moves out of the way.
So an apple that spawns inside a loop of the snake is not a trap if the loop is roomy
enough to eat it and wait.

Often two or three moves are **equally best** (e.g. an apple diagonally ahead: going
up-then-right or right-then-up are the same length). Two models can both make "best
moves" and still take different paths; the label says "best move (1 of 2 tied)".

| Move | Grade |
|---|---|
| Best move | 1.0 |
| Safe path to the apple, but 2 / 4 / ... moves slower | 0.5 / 0.33 / ... |
| Stays alive but ignores a safe path to the apple | 0.4 |
| Eats by a path it can't escape from afterwards (greedy) | 0.2 |
| Doomed: can't escape at all | 0.1 or less |
| Dies immediately | 0.0 |

Colour is a running average of these grades, stretched so random play is red and only
consistent best play is bright green. **The brain never sees any of this.**

## Senses (what the brain sees)

Each model is trained with one kind of senses and keeps it; the compare window can
put models with different senses side by side.

**basic (11 inputs)** - danger in the 3 squares next to the head, direction, and which
way the apple is. It can't see a trap coming, so it plateaus around 27-32 apples.

**vision (42 inputs)** - everything relative to the way the head faces, and
*time-aware*: a body cell only counts as blocking if it will still be there when the
head arrives, because the tail keeps moving away.

- For each of the 3 moves: dies instantly? how much room is left (vs free cells and
  vs its own length)? can it still reach its own tail? how far is the apple along the
  real shortest path?
- 8 lines of sight: distance to wall, to the first body cell that will still be there,
  and to the apple if it's on that line.
- Where the apple is (ahead/behind, left/right) and the snake's length.

**vision2 (51 inputs, the default)** - vision plus a look-ahead for each move:
can the snake escape after it (including waiting it out inside its own loop), how long
until its body opens a way, and whether racing to the apple by the shortest path and
eating it still leaves a way out (it simulates the whole path with the tail following).

**vision3 (57 inputs)** - vision2 plus tidiness for each move: how many
squares around the new head are wall or body (hugging keeps a long snake packed
neatly), and how much of the free board the move fences off. Snakes mostly die by
spiralling round the edge and fencing a big empty area off inside themselves; this
lets the brain see that happening.

**vision4 (66 inputs)** - vision3 plus, for each move: how close the head
stays to its own tail (tail-following can never trap a snake), whether the apple is still
reachable or has been fenced off, and how many pieces the free board is cut into. It
contains all of vision3, so it can't see less; whether it's better is for training to show.

**vision5 (81 inputs, the default)** - vision4 plus a *route*: one fixed loop through every
square of the board (a Hamiltonian cycle, `cycle.py`). A snake whose body lies along the
route in order can never trap itself, so this is how Snake is actually beaten: follow
the route, and take shortcuts towards the apple only when they keep the body in order
with room to spare before the tail. For each move the brain sees how far along the route
it jumps, whether it just follows the route, whether it's a safe shortcut, and the route
distance to the apple from there; plus overall: free route ahead, route distance to the
apple, and whether the body is in route order right now. Even board sizes only (odd x odd
boards have no such loop).

Proof that this is enough: a scripted player that reads only these senses (take the safe
move closest to the apple along the route; past half full, only follow) **filled the board
in 200/200 games on 10x10 and 100/100 on 20x20**, with no deaths at all. The brain has to
learn to do the same; a potential-based route reward (`ROUTE_WEIGHT`: keeping the body in
route order is worth 5, breaking it costs 5, restoring it earns it back) helps it connect
"broke the order" with "died 300 moves later".

**Route safety (vision5 option, on by default).** A pure learner with vision5 got stuck
around 70-75 of 97 on 10x10: learning that a shortcut kills you 300 moves later is very
hard. So a vision5 model can be given a rule: whenever a move exists that keeps the body
in route order with room to spare and doesn't skip past the apple, the brain may only
choose among those. That rule alone can neither crash nor starve, so it always fills the
board (the 10x10 test model went from 28/400 boards filled to 400/400 with it switched
on, untrained for it). What's left for the brain to learn is *speed*: which safe
shortcut gets to each apple fastest. Once 90% of games fill the board, training judges
progress on moves-to-fill instead of apples, and best.pt prefers the faster model.
Answer "no" to route safety to let the brain learn safety by itself.

These are facts about the board, not the referee's verdict: the brain still has to
learn what to do with them.

## Files

```
run.bat        double-click: menu (first run creates .venv and installs everything)
delete_models.bat  pick which trained models to delete (asks first)
menu.py        train new / continue training / compare models
train.py       command-line training   (python train.py --name x --rounds 20)
compare.py     command-line comparison (python compare.py modelA modelB)
snake/
  config.py    every tunable number
  game.py      N independent snake games stepped together with numpy
  senses.py    game state -> inputs: basic (11), vision (42) ... vision4 (66), vision5 (81)
  cycle.py     the route through every square (Hamiltonian cycle) used by vision5
  lookahead.py try every move out on copies of the board before choosing
  brain.py     the network: inputs -> hidden -> hidden -> 3 (turn left / straight / right)
  agent.py     replay memory, exploration, Double DQN training step
  paths.py     time-aware path finding: shortest paths, escaping, safe-to-eat checks
  cpu.py       how many CPU threads to use
  oracle.py    best-move referee (numba, multi-core)
  trainer.py   headless training in stages (short / mid / long), logging, saving
  arena.py     fair side-by-side comparison of models, records every move for replays
  render.py    the compare window: follow, pause, replay
  solo.py      one snake at full speed, with an end screen
  fast.py      the fast single-snake engine it uses
  models.py    models/<name>/ folders
  console.py   console output shared by the menu and scripts
tests/         game rules, fairness and referee tests
models/<name>/ best.pt, latest.pt, stageN-*.pt, positions.npz (saved starting positions),
               log.csv, info.json
```

## Results

All on a 20x20 board, trained with 1000 snakes, then compared on **identical games**
(500 snakes per model, same seeds, no random moves):

| Model | What changed | Avg apples | Best in training | Accuracy |
|---|---|---|---|---|
| quick-demo | basic senses (11 inputs) | 31.6 | 69 | 97.8% |
| vision-v1 | vision senses (42 inputs) | 84.4 | 152 | 96.0% |
| exp-b | + gamma 0.98, 0.1% random moves | 108.1 | | 96.7% |
| exp-c | + brain 256 wide | 111.1 | | 95.4% |
| **vision-v2** | exp-b + safe exploration (brain 128) | **113.6** | 227 | 95.7% |

### Round 2: 900 identical games per model (300 snakes x 3 rounds, 20x20, no random moves)

| Model | Senses | Learning | Avg | Median | Best | Rounds won | Sole best on |
|---|---|---|---|---|---|---|---|
| **V3-Full** | vision3 | 3-step + tidiness | **135.9** | 134 | 317 | 3 of 3 | 40% of seeds |
| V3-FULL-VISIONV2 | vision2 | 3-step + tidiness | 127.0 | 125 | 347 | 0 | 24% |
| V3 | vision2 | 1-step | 114.5 | 118 | 270 | 0 | 17% |
| vision-v2 | vision | 1-step | 115.6 | 118 | 195 | 0 | 17% |

- 3-step learning + tidiness: +12.5 apples (V3 -> V3-FULL-VISIONV2).
- vision3 over vision2: +9.0 apples on average (95% CI +5.9 to +12.0), but vision2 still
  wins 40% of individual games, so watching one snake can easily show the opposite.
- No model has filled the board yet; 90% of deaths are the snake boxing itself in.
- The referee's own strategy (always play its "best move") averages only 80 apples on
  the same kind of games. The brains have outgrown it, so accuracy is no longer a useful
  target: score is.

vision-v2 won every round against the others and was the sole best snake on the most
seeds. A 256-wide brain added little, so the default stays at 128. Accuracy looks flat
because better snakes live longer and spend more time in the hard, crowded late game.

## Recommended settings for the best model

All defaults, so in the menu press Enter through every question:

| Setting | Value |
|---|---|
| Senses | vision5 (needs an even grid size) |
| Route safety | on (fills the board; the brain learns to do it fast) |
| Look-ahead | on |
| Train in stages | on (short 0-50, mid 50-150, long 150-full) |
| Snakes / grid | 300 / 20 (with route safety; 1000 on 10x10 or 12x12) |
| Maximum rounds per stage | 200 (moves on after 15 rounds without +1%, or once mastered) |
| CPU | every thread |

With route safety every game lasts until the board is full, and filling 20x20 takes
around 20,000-40,000 moves, so long-stage rounds are slow: roughly 15 minutes each with
1000 snakes. Use about 300 snakes on 20x20 (a few minutes per round), or train on 10x10 /
12x12 first, where a whole run takes 10-20 minutes.

### 10x10 test with vision5 (1000 snakes, ~10 minutes of training each)

| Setup | Boards filled | Moves per fill |
|---|---|---|
| vision5, learning safety by itself | 28 / 400 (avg 70 of 97) | 1,471 |
| same model, route safety switched on afterwards | 400 / 400 | 2,144 |
| **vision5 trained with route safety (demo-10x10-fills)** | **300 / 300** | **1,979** |
| scripted: route + greedy safe shortcuts | 300 / 300 | 2,080 |
| scripted: route + shortcuts only while under half full | 300 / 300 | 1,473 |

The trained model always fills the board and is faster than greedy shortcuts, but not yet
as fast as the best simple rule; more training (and the 20x20 run) is where that's won.

The compare window and its console summary now show each model's best game and median as
well as the average, how many boards it filled, and the average moves it took to fill one.

## Next steps (not built yet)

- Charts from `log.csv`, GIF of round 1 vs a trained round, README.
- Git repo.
