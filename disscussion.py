import numpy as np
import random
import torch
from torch import nn
from torch import optim

BATCH_SIZE = 32
CAPACITY = 10000


class Discussion:

    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions
        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(num_states, 32))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(32, 32))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))

        print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
