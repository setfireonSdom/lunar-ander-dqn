from itertools import count
import numpy as np
import torch
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, env, agent, n_episode):
        self.env = env
        self.n_episode = n_episode
        self.agent = agent
        self.batch_size = 64
        # self.losslist = []
        self.rewardlist = []

    # 获取当前状态，将env返回的状态通过transpose调换轴后作为状态
    def get_state(self, obs):
        state = torch.from_numpy(obs)
        return state.unsqueeze(0)  # 转化为四维的数据结构

    def epsilon_decay(self,initial_epsilon, episode, decay_rate):
        self.agent.DQN.eps = initial_epsilon * (1 / (1 + decay_rate * episode))

    # 训练智能体
    def train(self):
        total_step = 0
        decay = 0.0032
        for episode in range(1,self.n_episode+1):
            self.epsilon_decay(1,episode,decay)
            obs = self.env.reset()[0]
            state = self.get_state(obs)
            episode_reward = 0.0
            # print('episode:',episode)
            for t in count():
                # print(state.shape)
                # action是个int
                action = self.agent.DQN.select_action(state.to(self.agent.DQN.device))

                total_step += 1
                obs, reward, done, tru, info = self.env.step(action)
                episode_reward += reward

                if not done and not tru:
                    next_state = self.get_state(obs)
                else:
                    next_state = None
                # print(next_state.shape)
                reward = torch.tensor([reward], device=self.agent.DQN.device).to(torch.float32)

                self.agent.DQN.mm.push((state, action, next_state, reward.to('cpu')))  # 里面的数据都是Tensor
                state = next_state
                # 经验池满了之后开始学习
                if len(self.agent.DQN.mm.memory) > self.batch_size:
                    self.agent.learn(self.batch_size)
                    if total_step % 50 == 0:
                        self.agent.update_param()

                if done or tru:
                    break

            if episode % 20 == 0:
                torch.save(self.agent.DQN.state_dict(),"{}_episode.pth".format("Lander"))
                avg_reward = np.mean(self.rewardlist[-20:])
                print('Total steps: {} \t Episode: {}/{} \t pre-20-epoch avg-reward: {}'.format(total_step, episode, self.n_episode,avg_reward))

            self.rewardlist.append(episode_reward)
        return

    # 绘制奖励曲线
    def plot_reward(self):

        plt.plot(self.rewardlist)
        plt.xlabel("episode")
        plt.ylabel("episode_reward")
        plt.title('train_reward')

        plt.show()
    def save_array(self):
        np.save('target-array.npy', self.rewardlist)
        self.env.close()