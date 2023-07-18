import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym

# Define the Actor-Critic network architecture
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

# A2C agent class
class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.actor_critic = ActorCritic(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs, _ = self.actor_critic(state)
        action = torch.multinomial(action_probs, num_samples=1).item()
        return action

    def update(self, log_probs, values, rewards, masks, gamma=0.99, tau=0.95):
        returns = []
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G * masks[t]
            returns.insert(0, G)

        returns = torch.tensor(returns)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        total_loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

# Main training loop
def train_a2c(agent, num_episodes=800, max_steps_per_episode=500):
    Rli = []
    for episode in range(num_episodes):
        state = agent.env.reset()[0]
        log_probs = []
        values = []
        rewards = []
        masks = []
        R = 0
        for t in range(max_steps_per_episode):
            action_probs, state_value = agent.actor_critic(torch.FloatTensor(state))
            action = torch.multinomial(action_probs, num_samples=1).item()

            next_state, reward, done,t, _ = agent.env.step(action)

            log_prob = torch.log(action_probs.squeeze(0)[action])
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            masks.append(1 - done)

            state = next_state

            if done:
                break

        agent.update(log_probs, values, rewards, masks)
        print(f"Episode {episode}, Total Reward: {sum(rewards)}")
        Rli.append(sum(rewards))
        # if episode % 10 == 0:
        #     print(f"Episode {episode}, Total Reward: {sum(rewards)}")
    torch.save(agent.actor_critic.state_dict(),'a2c_cartpole.pth')
    import numpy as np
    np.save('a2c_cp.npy',Rli)

# Example usage
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = A2CAgent(env)
    train_a2c(agent)
