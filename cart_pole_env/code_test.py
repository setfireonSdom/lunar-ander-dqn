import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

def reinforce(env, policy_net, num_episodes, lr):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    for episode in range(num_episodes):
        log_probs = []
        rewards = []

        state = env.reset()[0]
        done = False
        R = 0
        while not done:
            state_tensor = torch.FloatTensor([state])
            action_probs = policy_net(state_tensor)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

            next_state, reward, done,t,_ = env.step(action.item())
            rewards.append(reward)
            R+=reward
            state = next_state

        returns = calculate_returns(rewards)
        policy_loss = calculate_policy_loss(log_probs, returns)

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Policy Loss: {policy_loss.item()}")
            print(R)
    torch.save(policy_net.state_dict(),'cart_pole_model.pth')

def calculate_returns(rewards):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.9 * G  # Discounted return with discount factor 0.9
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Normalize returns
    return returns

def calculate_policy_loss(log_probs, returns):
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)
    policy_loss = torch.cat(policy_loss).sum()
    return policy_loss

# Example usage
env = gym.make('CartPole-v0')
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = PolicyNetwork(input_dim, output_dim)
reinforce(env, policy_net, num_episodes=1000, lr=0.001)