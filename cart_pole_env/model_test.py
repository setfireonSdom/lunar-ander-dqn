import gymnasium as gym
env = gym.make('CartPole-v0',render_mode='human')
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

model = PolicyNetwork(env.observation_space.shape[0],env.action_space.n)
model.load_state_dict(torch.load('cart_pole_model.pth'))

for episode in range(100):

    state = env.reset()[0]
    done = False
    t = False
    R = 0
    while not done and not t:
        state_tensor = torch.FloatTensor([state])
        action_probs = model(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()


        next_state, reward, done, t, _ = env.step(action.item())
        R += reward
        state = next_state

    if (episode + 1) % 10 == 0:
        print(R)