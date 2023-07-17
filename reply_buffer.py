import random
from collections import deque

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, data):
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)  # 从经验池中随机采样

    def __len__(self):
        return len(self.memory)