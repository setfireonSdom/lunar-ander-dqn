import gymnasium as gym

env = gym.make('LunarLander-v2',render_mode='human')
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F


class Pg(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Pg, self).__init__()
        self.l1 = nn.Linear(input_dim,128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128,output_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        # 我没有加dim
        return F.softmax(x,dim=1)

model = Pg(env.observation_space.shape[0],env.action_space.n)
model.load_state_dict(torch.load('Pg_lander.pth',map_location=torch.device('cpu')))

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

# import matplotlib.pyplot as plt
# import numpy as np
# data = np.load('pglander.npy')
# plt.plot(data)
# plt.show()