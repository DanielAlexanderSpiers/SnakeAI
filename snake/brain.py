"""The neural network: game inputs in, one predicted future reward (Q-value) per move out."""

from torch import nn

N_ACTIONS = 3   # turn left, straight, turn right


class Brain(nn.Module):
    def __init__(self, n_inputs, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)
