# import matplotlib.pyplot as plt
# import numpy as np
# data = np.load('a2c_cp.npy')
# plt.plot(data)
# plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc_shared = nn.Linear(input_dim, 64)
        self.fc_actor = nn.Linear(64, output_dim)
        self.fc_critic = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc_shared(x))
        action_probs = F.softmax(self.fc_actor(x), dim=-1)
        state_value = self.fc_critic(x)
        return action_probs, state_value


def test_a2c(env, model,num_episodes=100, max_steps_per_episode=500):
    Rli = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        log_probs = []
        values = []
        rewards = []
        masks = []
        R = 0
        for t in range(max_steps_per_episode):
            action_probs, state_value = model(torch.FloatTensor(state))
            action = torch.multinomial(action_probs, num_samples=1).item()

            next_state, reward, done,t, _ = env.step(action)

            rewards.append(reward)

            state = next_state

            if done:
                break

        print(f"Episode {episode}, Total Reward: {sum(rewards)}")


env = gym.make('CartPole-v1',render_mode='human')
model = ActorCritic(env.observation_space.shape[0],env.action_space.n)
model.load_state_dict(torch.load('a2c_cartpole.pth'))
test_a2c(env,model)