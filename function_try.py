# 1.
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import torch
# data = torch.range(-100,100,1)
# print(data)
# output = F.relu(data)
# print(output)
# plt.plot(data,output)
# plt.show()

# 2.
# import torch.nn as nn
# import torch
# m = nn.Linear(8,512)
# data = torch.rand(8)
# print(data.shape)
# print(m(data).shape)

# # 3.
# def epsilon_decay(initial_epsilon, episode, decay_rate):
#     eps = initial_epsilon * (1 / (1 + decay_rate * episode))
#     return eps
#
# ipl_eps = 1
# eps1 = 1
# eps2 = 1
# eps3 = 1
# # decay越小eps降低的越慢
# decay_rate1 = 0.1
# decay_rate2 = 0.01
# # o.0032不错
# decay_rate3 = 0.0035
#
# eps_li1 = []
# eps_li2 = []
# eps_li3 = []
# for i in range(1000):
#     eps_li1.append(eps1)
#     eps_li2.append(eps2)
#     eps_li3.append(eps3)
#     eps1 = epsilon_decay(ipl_eps,i,decay_rate1)
#     eps2 = epsilon_decay(ipl_eps, i, decay_rate2)
#     eps3 = epsilon_decay(ipl_eps, i, decay_rate3)
#
# import matplotlib.pyplot as plt
# plt.plot(eps_li1,label='1')
# plt.plot(eps_li2,label='2')
# plt.plot(eps_li3,label='3',color='black')
# plt.legend()
# plt.show()

# 4
# import torch
# from dqn_model import DQN
# from reply_buffer import ReplayMemory
# mm = ReplayMemory(200000)
# model = DQN(mm)
# output = model(torch.from_numpy(env.reset()[0]))
# print(output)

# 5.
# import numpy as np
# data = np.load('array.npy')
# import matplotlib.pyplot as plt
# plt.plot(data)
# plt.show()

# 6
# import time
# now = time.time()
# end = time.time()
# time.sleep(5)
# cost = end - now
# print(cost/60)
# for i


# 7 用来做对比，完全随机的玩LunarLander
# import gymnasium as gym
# from dqn_model import DQN
# import torch
# model = DQN(None)
# model.load_state_dict(torch.load('double-dqn-Lander_episode.pth', map_location=torch.device('cpu')))
# env = gym.make('LunarLander-v2',render_mode='human')
# reward_li = []
# for i in range(10):
#     s = env.reset()[0]
#     done = False
#     t = False
#     R = 0
#     while not done and not t:
#         a = model(torch.from_numpy(s).unsqueeze(0)).max(1)[1].view(1, 1).to('cpu').item()
#         s,r,done,t,_ = env.step(a)
#         R+=r
#     reward_li.append(R)
#     print(i,R)
# import numpy as np
# import matplotlib.pyplot as plt
# plt.plot(reward_li)
# plt.plot([np.mean(reward_li)]*10)
# plt.show()

# 8。
import gymnasium as gym
from dqn_model import DQN
env = gym.make('LunarLander-v2',render_mode='human')
reward_li = []
for i in range(5000):
    s = env.reset()[0]
    done = False
    t = False
    R = 0
    while not done and not t:
        s,r,done,t,_ = env.step(env.action_space.sample())
        R+=r
    reward_li.append(R)
# import numpy as np
# import matplotlib.pyplot as plt
# plt.plot(reward_li)
# plt.plot([np.mean(reward_li)]*5000)
# plt.savefig("random-2.jpg")
# plt.show()

# 9.
# import numpy as np
# data= np.load('target-array-5.npy')
# import matplotlib.pyplot as plt
# plt.plot(data,label='Target REWARD')
# plt.plot([np.max(data)]*len(data),label="MAX VALUE:{}".format(np.max(data)))
# # plt.ylim(-600, 400)
# plt.legend()
# # plt.yticks(np.arange(-1000, 500, 100))
# plt.savefig('target-lander_model-5+1.jpg')
# plt.show()


# 10.
# def calculate_average_per_ten(input_list):
#     result = []
#     for i in range(0, len(input_list), 10):
#         sublist = input_list[i:i+10]  # 获取当前十个元素的子列表
#         average = sum(sublist) / len(sublist)  # 计算子列表的平均值
#         result.append(average)  # 将平均值添加到结果列表中
#     return result
# import numpy as np
# x1 = np.load('no-target-array1.npy')
# x2 = np.load('target-array-5.npy')
# x3 = np.load('double-dqn-array.npy')
# print(np.mean(x1),np.mean(x2),np.mean(x3))
# import matplotlib.pyplot as plt
# p1 = calculate_average_per_ten(x1)
# p2 = calculate_average_per_ten(x2)
# p3 = calculate_average_per_ten(x3)
# plt.plot(p1,label="no-target")
# plt.plot(p2,label="target-dqn")
# plt.plot(p3,label="double dqn")
# plt.legend()
# plt.savefig("Comparison-tree.jpg")
# plt.show()

