# SnakeAI

**A neural network that teaches itself to play Snake.** It is never shown a correct move:
a thousand snakes play at once, every move is scored with a reward, and one shared network
learns from all of that experience which moves lead to more apples and fewer deaths. Over six
generations of what the network can sense, it went from averaging 29 apples to filling the
entire 20x20 board.

![Four generations of the network playing identical games](docs/compare-generations.png)
*Four trained generations playing the exact same 100 games. The first network (top left) has
lost every snake; each newer one keeps more alive, and V5 (bottom right) hasn't lost any.*

## The network

- **A Deep Q-Network (DQN) in PyTorch.** The snake's senses go in; out come three numbers, the
  network's estimate of the future reward of turning left, going straight or turning right.
  The snake takes the move with the highest estimate. Architecture: inputs -> 128 -> 128 -> 3.
- **Reinforcement learning, not labelled data.** +10 for an apple, +100 for filling the board,
  -10 for crashing, -20 for starving. The network learns to predict those rewards, and in
  doing so learns to play.
- **1000 snakes, one brain.** Every snake is driven by the same network and every move goes into
  a shared replay memory of 300,000 experiences, so each tick of the game is 1000 lessons. The
  network trains on random batches from that memory.
- **The standard DQN toolkit:** Double DQN with a slowly-following target network, 3-step returns
  (a death is felt by the moves that set it up, not only the last one), epsilon-greedy
  exploration that never picks an instantly fatal random move, Huber loss, Adam.
- **Look-ahead at play time:** before each move the snake tries all three moves on copies of the
  board and lets the network judge where each one leads.

## What training looks like

Training runs headless and flat out on the CPU. It moves through three stages (short games to 50
apples, mid games to 150, then long games until the board is full) so the network gets plenty of
practice at the late game, where snakes actually die. Each stage ends by itself when the network
stops improving.

![Console output while training v4](docs/training-console.png)

Every model keeps a log of every round. Plotted, those logs show each generation learning more
than the one before:

![Learning curves of every generation](docs/learning-curves.png)

## Six generations of senses

The biggest lever was never the size of the network: it was **what the network can see**. Each
generation adds inputs that describe the board better; all of them are facts about the board,
never "the best move".

| Generation | Inputs | What the network can see | Average apples |
|---|---|---|---|
| basic | 11 | Danger in the 3 squares next to the head, its direction, which way the apple is | 29 |
| vision | 42 | How much room each move leaves, whether it can still reach its own tail, the real path distance to the apple, 8 lines of sight; all time-aware (it knows its tail moves out of the way) | 116 |
| vision2 | 51 | Whether it can escape after each move, including waiting inside a loop of its own body, and whether it can eat the apple and still get out | 127 |
| vision3 | 57 | Tidiness: hugging walls and its own body, how much space a move would fence off | 136 |
| vision4 | 66 | Distance to its own tail, whether the apple has been fenced off, how many pieces the free board is split into | 144 |
| vision5 | 81 | A fixed route through every square of the board (a Hamiltonian cycle): whether a move follows it, whether a shortcut off it is safe, how far the apple is along it | **397: fills the board** |

The networks up to vision4 all hit the same wall at around a third of the board: they would take
a shortcut that boxed them in 300 moves later, and that consequence is too far away to learn
from reward alone. vision5 gives the network the idea of a route; a *route safety* layer then
only lets it choose moves that keep its body in route order, and the network's job becomes
choosing which safe shortcut reaches each apple fastest.

## Using the trained networks

![V5 against the first network on the same 150 games](docs/compare-v5-vs-basic.png)
*The first network (left) against V5 on the same 150 games: every basic snake has died, every V5
snake is still growing.*

The **compare window** plays up to four trained networks side by side on *identical* games: every
snake starts in the same place and snake #N gets the same apples on every board, so the only
difference is the network. Tab follows one snake across every board, R replays the last 10 seconds
of its life on every board at once, and results show average, best, median and boards filled.
(Colours show how often a move agrees with a simple best-move referee; V5 follows its own route,
so it shows yellow while never dying.)

The **solo viewer** runs one network on one snake with a dedicated fast engine (about 24,000
moves per second), and stops on an end screen when the board is full:

![V5 has filled the board](docs/v5-board-filled.png)

## Results

20x20 board, no random moves. Averages from side-by-side comparisons on identical games (basic:
its training average); best = best single game recorded.

| Model | Senses | Average apples | Best game | Boards filled |
|---|---|---|---|---|
| Snakey | basic | 29 | 80 | 0% |
| vision-v2 | vision | 116 | 195 | 0% |
| V3-Full | vision3 | 136 | 317 | 0% |
| v4 | vision4 + look-ahead | 144 | 347 | 0% |
| **V5** | **vision5 + look-ahead + route safety** | **397** | **397** | **100%** |

## Try it (Windows)

1. Install Python 3.12 or newer.
2. Double-click **`run.bat`**. The first run sets up a virtual environment with PyTorch, numpy,
   numba and pygame (a few hundred MB, once).
3. The trained networks are included, so you can go straight to **Compare models** or **Watch
   one snake at full speed**. **Train a new model** trains your own; `train_best.bat` uses the
   best settings found.

## Project layout

```
run.bat, menu.py          menu: train, continue training, compare, watch, delete
train.py, compare.py      command-line training and comparing
snake/
  brain.py                the neural network
  agent.py                replay memory, Double DQN training step, n-step returns
  trainer.py              staged training loop
  senses.py               what the network sees: basic, vision ... vision5
  game.py                 1000 snake games stepped together with numpy
  lookahead.py            trying moves out before choosing, route safety
  cycle.py, paths.py      the route through every square, time-aware path finding
  oracle.py               the best-move referee used for colours and the accuracy stat
  arena.py, render.py     side-by-side comparisons, follow mode, replays
  solo.py, fast.py        the one-snake viewer and its fast engine
  config.py               every setting (override any with SNAKEAI_CONFIG=NAME=value,...)
models/                   the trained networks and their training logs
tests/                    game rules, senses, route, fast engine
```
