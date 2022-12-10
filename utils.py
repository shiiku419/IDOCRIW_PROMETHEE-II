import torch
import numpy as np
from collections import namedtuple
from environment import Environment
# from model import QNet

Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

gamma = 0.99
batch_size = 32
lr = 0.0001
initial_exploration = 1000
goal_score = 200
log_interval = 1
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Multi_Step
n_step = 1

# PER
small_epsilon = 0.0001
alpha = 1
beta_start = 0.1

# Noisy Net
sigma_zero = 0.5

# Distributional
num_support = 8
V_max = 5
V_min = -5


class Memory(object):
    def __init__(self, capacity):
        self.env = Environment()
        self.memory = [[] for _ in range(self.env.n_member)]
        self.memory_probabiliy = [[] for _ in range(self.env.n_member)]
        self.capacity = capacity
        self.position = 0
        self.reset_local()

    def reset_local(self):
        self.local_step = 0
        self.local_state = None
        self.local_action = None
        self.local_rewards = [[] for _ in range(self.env.n_member)]

    def push(self, state, next_state, action, reward, mask, id):
        self.local_step += 1
        self.local_rewards[id].append(reward)
        if self.local_step == 1:
            self.local_state = state
            self.local_action = action
        if self.local_step == n_step:
            reward = 0
            for idx, local_reward in enumerate(self.local_rewards[id]):
                reward += (gamma ** idx) * local_reward
            self.push_to_memory(self.local_state, next_state,
                                self.local_action, reward, mask, id)
            self.reset_local()
        if mask == 0:
            self.reset_local()

    def push_to_memory(self, state, next_state, action, reward, mask, id):
        if len(self.memory[id]) > 0:
            max_probability = max(self.memory_probabiliy[id])
        else:
            max_probability = small_epsilon

        if len(self.memory[id]) < self.capacity:
            self.memory[id].append(Transition(
                state, next_state, action, reward, mask))
            self.memory_probabiliy[id].append(max_probability)
        else:
            self.memory[id][self.position] = Transition(
                state, next_state, action, reward, mask)
            self.memory_probabiliy[id][self.position] = max_probability

        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.memory)
