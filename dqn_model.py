import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, mm,n_actions=4):
        super(DQN, self).__init__()
        self.actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.bn1 = nn.BatchNorm2d(32)
        self.l1 = nn.Linear(8,64)
        # self.bn2 = nn.BatchNorm2d(64)
        self.l2 = nn.Linear(64,64)
        # self.bn3 = nn.BatchNorm2d(64)
        self.l3 = nn.Linear(64, n_actions)
        self.eps = 1
        self.gamma = 0.99
        # replay buffer
        self.mm = mm
        self.optimizer = optim.Adam(self.parameters(), lr=0.0005)
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def select_action(self, state):
        if random.random() < self.eps:
            action = torch.tensor([[random.randrange(self.actions)]], device=self.device, dtype=torch.long)
        else:
            action = self(state).detach().max(1)[1].view(1, 1).to(self.device)
        return action.item()

    def epsilon_decay(self,initial_epsilon, episode, decay_rate):
        self.eps = initial_epsilon * (1 / (1 + decay_rate * episode))