import gym.spaces
import numpy as np
import matplotlib as plt
from collections import namedtuple
import random
import math
from scipy.special import softmax
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = [[] for _ in range(7)]
        self.index = 0

    def push(self, state, action, state_next, reward, agent_id):

        if len(self.memory[agent_id]) < self.capacity:
            self.memory[agent_id].append(None)

        self.memory[agent_id][self.index] = Transition(
            state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
