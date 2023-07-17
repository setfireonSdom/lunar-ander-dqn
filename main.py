# 一个回合能达到至少200分，则认为这个回合是一个solution.
# 动作空间：4，状态空间：(8,)
# 状态的初始shape：numpy.ndarray
# 初始数据类型：float32
# env.step()的输入:python int,numpy int
import gymnasium as gym
env = gym.make('LunarLander-v2')

from dqn_model import DQN
from dqn_agent import DQNAgent
from env_trainer import Trainer
from reply_buffer import ReplayMemory
memory = ReplayMemory(100000)
model = DQN(memory)
target_model = DQN(None)
agent = DQNAgent(model,target_model)
agent.update_param()
whole_env = Trainer(env,agent,2200)
import time
start = time.time()
whole_env.train()
end = time.time()
cost = end-start
whole_env.save_array()
print("Total time：{}分钟".format(cost/60))