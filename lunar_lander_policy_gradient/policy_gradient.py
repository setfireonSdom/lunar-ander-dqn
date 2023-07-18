# 与code_test.py里面的比较，
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from collections import deque
import numpy as np

class Pg(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Pg, self).__init__()
        self.l1 = nn.Linear(input_dim,128)
        self.l2 = nn.Linear(128,128)
        self.l3 = nn.Linear(128,64)
        self.l4 = nn.Linear(64,output_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        # 我没有加dim
        return F.softmax(x,dim=1)


def reinforce(env,max_epochs,policy):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = optim.Adam(policy.parameters(),lr=0.001)
    Rli = []
    for i in range(max_epochs):
        rewards = []
        R = 0
        log_probs = []
        s = env.reset()[0]
        done = False
        t = False
        while not done and not t:
            s = torch.FloatTensor([s]).to(device)
            a_prob = policy(s)
            a_li = Categorical(a_prob)
            a = a_li.sample()
            log_prob = a_li.log_prob(a)
            log_probs.append(log_prob.to(device))
            s_n,r,done,t,_ = env.step(a.item())
            rewards.append(r)
            R += r
            s = s_n
        returns = get_returns(rewards)
        returns.to(device)
        loss = policy_loss(returns,log_probs)
        opt.zero_grad()
        loss.backward()
        opt.step()
        Rli.append(R)
        print(R)
    #     if (i+1) % 10==0:
    #         print(f"前10/{i+1}回合的奖励累积和是: {sum(Rli[-10:])/10}")
    # torch.save(policy.state_dict(),'pglander.pth')
    # np.save('lander.npy',Rli)


def get_returns(rewards):
    G = 0
    returns = deque()
    for r in reversed(rewards):
        # 我没有加0.9这个折扣因子
        G = r + 0.9*G
        # 而且我的G添加到列表的方式错了，没有跟logp对应，我之前的方式是
        # 倒数第二个状态的G对应第一个log_prob,这样就不对了，第一个log_prob对应的应该是第一个G。
        returns.appendleft(G)
    returns = torch.tensor(returns)
    returns = (returns-returns.mean())/(returns.std()+1e-9)
    return returns

def policy_loss(returns,log_probs):
    loss = []
    for r,prob in zip(returns,log_probs):
        loss.append(-prob*r)
    loss = torch.cat(loss).sum()
    return loss


if __name__ == '__main__':
    # env = gym.make('CartPole-v0') LunarLander-v2 MountainCar-v0 Pendulum-v1 CarRacing-v2
    # Acrobot-v1
    # 使用'CartPole-v0'是没有什么问题的，我用我自己写的这块reinforce算法跑cartpole，没问题。
    # 当环境换成了'LunarLander-v2'，要训练很久很久还不行，为什么？
    env = gym.make('Acrobot-v1')

    policy = Pg(env.observation_space.shape[0],env.action_space.n)
    # print(env.action_space.n) 4
    # print(env.observation_space.shape) 8
    reinforce(env,2000,policy)