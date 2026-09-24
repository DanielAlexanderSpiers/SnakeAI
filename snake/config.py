"""Every tunable number in one place."""

# --- Board -------------------------------------------------------------------
DEFAULT_SNAKES = 1000
DEFAULT_GRID = 20
START_LENGTH = 3

# --- Rewards -----------------------------------------------------------------
REWARD_APPLE = 10.0
REWARD_WIN = 100.0            # filled every square of the board: the goal
REWARD_DEATH = -10.0          # hit a wall or its own body
REWARD_STARVE = -20.0         # ran out of moves before reaching the apple
REWARD_CLOSER = 0.1           # stepped closer to the apple
REWARD_FURTHER = -0.15        # stepped further away (slightly worse than closer is good)
# The closer/further nudge fades out as the snake fills the board: a long snake often
# has to take the long way round to stay safe, and shouldn't be punished for it.
SHAPING_FADES = True
# Tidiness: moves that fence off part of the board from the head cost a little and moves
# that open it up again earn the same back (potential-based shaping, so the best way to
# play is unchanged; it only teaches neat packing sooner). 0 turns it off.
COMPACT_WEIGHT = 3.0

# --- Starvation --------------------------------------------------------------
# Each new apple gives the snake a move budget of the larger of
#     manhattan distance to the apple + STARVE_SLACK + STARVE_SLACK_PER_SEGMENT * length
#     STARVE_AREA_FACTOR * squares on the board
# so a snake always has time to go all the way round the board to stay safe, and
# starvation only catches snakes that loop forever.
STARVE_SLACK = 10
STARVE_SLACK_PER_SEGMENT = 1
STARVE_AREA_FACTOR = 1.0

# --- Deep Q-Learning ---------------------------------------------------------
HIDDEN = 128
GAMMA = 0.98                  # how far ahead rewards count (0.98 beat 0.95: traps matter ~10+ moves later)
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
REPLAY_CAPACITY = 300_000
WARMUP_TRANSITIONS = 5_000    # collect this much experience before training starts
TRANSITIONS_PER_UPDATE = 1024 # one training step per this many new experiences (1024 trained 2.6x faster than 512 and scored higher)
TARGET_TAU = 0.005            # soft update rate of the target network
# Learn from the next N_STEP moves at once instead of one. A death then reaches the
# moves that set it up N_STEP moves earlier, which is what lets a snake see traps coming.
N_STEP = 3
GRAD_CLIP = 10.0

EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY_TRANSITIONS = 400_000   # linear decay from EPS_START to EPS_END
# Random exploration moves only pick moves that don't kill the snake on the spot (when
# one exists). The brain still decides everything else itself; it just stops losing
# long games to pointless random suicides, so it gets far more late-game practice.
SAFE_EXPLORATION = True

# --- CPU -----------------------------------------------------------------------
# Threads used for path-finding and the brain. 0 = all of them (fastest, PC feels busy),
# -2 = all but two (a little slower, PC stays responsive), 4 = exactly four. Default: all.
CPU_THREADS = 0

# --- Route (vision5) -------------------------------------------------------------
# A shortcut off the route is only "safe" if at least this many free route squares stay
# between the new head and the tail (room to grow while eating).
CYCLE_BUFFER = 4
# Route reward (vision5 models only): keeping the body in route order is worth this much,
# breaking the order costs it, getting back in order earns it back (potential-based, like
# the tidiness reward, so it doesn't change the best way to play; it just teaches it sooner).
ROUTE_WEIGHT = 5.0
# Route safety (vision5 models, asked when creating one): whenever at least one move keeps
# the body in route order with room to spare, the brain may only choose among those. This
# alone provably fills the board; the brain decides which safe move (how fast it fills).
ROUTE_SHIELD = True

# --- Training stages -----------------------------------------------------------
MASTERED = 0.97               # a stage also ends once its 5-round average gets this close to the
                              # stage's top (e.g. 48.5 of 50): there's nothing left to learn there
EARLIER_SHARE = 0.3           # in the long stage, this share of games starts from mid-length
                              # positions instead, so the mid game isn't unlearned
# Training runs in stages by length: short games (0-50 apples), then mid (50-150), then
# long (150 until the board is full). Each stage ends when it stops improving or hits
# its round limit. Mid and long games start from real positions the model reached
# earlier. STAGES lists the apple counts where one stage hands over to the next.
CURRICULUM = True
STAGES = "50,150"
CONTROL_SHARE = 0.2           # in mid/long stages this share of snakes plays whole games from the
                              # start, so nothing learned earlier is forgotten and full-game
                              # scores are still measured
POOL_SIZE = 3000              # saved starting positions kept per stage
POOL_MIN = 100                # fewer than this and the stage tops its pool up before starting
STAGE_EPS_START = 0.02        # a little fresh exploration at the start of each later stage...
STAGE_EPS_DECAY = 200_000     # ...fading to EPS_END over this many moves
LOOKAHEAD = True              # new models try every move out before choosing (lookahead.py)

# --- Training run -------------------------------------------------------------
DEFAULT_ROUNDS = 200          # upper limit; training normally stops itself earlier (below)
PATIENCE_ROUNDS = 15          # stop once the 5-round average score hasn't improved for this many rounds
MIN_IMPROVEMENT = 0.01        # ...by at least 1%
GRADE_SAMPLE = 64             # while training, referee grades this many snakes per tick (for the accuracy stat)
SAVE_EVERY_ROUNDS = 5

# --- Move-quality colouring (never shown to the AI) --------------------------
# Grades the referee gives (1.0 = best move, 0.0 = dies immediately). See oracle.py.
GRADE_STALL = 0.4             # stays alive, but ignored a safe path to the apple
GRADE_GREEDY = 0.2            # grabs the apple by a path that boxes the snake in afterwards
GRADE_TRAPPED = 0.1           # can reach neither the apple nor its own tail: doomed
QUALITY_EMA = 0.12            # how quickly a snake's colour reacts to new moves
# Random moves still hit a best move ~half the time (often two moves are equally good),
# so raw quality sits around 0.7 even for a clueless snake. Colour uses quality ** power
# so random play shows red and only consistently-best play shows bright green. With
# these numbers one boxing-in move turns a green snake orange for a few moves.
COLOUR_POWER = 8

# --- Compare window ------------------------------------------------------------
DEFAULT_COMPARE_SNAKES = 200  # per model
PANEL_WIDTH = 360
FPS = 60
SPEEDS = [2, 5, 10, 20, 40, 80, 160, 320, 0]   # ticks per second, 0 = as fast as possible
DEFAULT_SPEED_INDEX = 3

# --- Experiments ---------------------------------------------------------------
# Override any setting above for one run without editing this file, e.g.
#   set SNAKEAI_CONFIG=GAMMA=0.98,EPS_END=0.001        (Windows cmd)
# Each model records the settings it was trained with in its info.json.
import os as _os

for _item in filter(None, _os.environ.get("SNAKEAI_CONFIG", "").split(",")):
    _key, _value = (p.strip() for p in _item.split("=", 1))
    if isinstance(globals()[_key], bool):
        globals()[_key] = _value.lower() in ("1", "true", "yes", "on")
    else:
        globals()[_key] = type(globals()[_key])(_value)
