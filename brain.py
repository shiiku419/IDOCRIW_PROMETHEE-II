import numpy as np
import random
import torch.nn as nn
import torch
import torch.optim as optim
from model import QNet
from utils import Memory, batch_size, device, small_epsilon, alpha
from environment import Environment
from collections import namedtuple

Transition = namedtuple(
    'Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Brain:

    def __init__(self, num_inputs, num_actions, BATCH_SIZE=32, CAPACITY=10000, GAMMA=0.99):
        self.num_actions = num_actions

        self.BATCH_SIZE = BATCH_SIZE
        self.CAPACITY = CAPACITY
        self.GAMMA = GAMMA

        self.memory = Memory(CAPACITY)
        self.env = Environment()

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(7, 36))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(36, 36))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(36, 7*10))

        # print(self.model)

        self.online_net = QNet(num_inputs, num_actions)
        self.target_net = QNet(num_inputs, num_actions)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=0.001)

    def train(self, epsilon, beta, id):
        epsilon -= 0.00005
        epsilon = max(epsilon, 0.1)
        beta += 0.00005
        beta = min(1, beta)

        batch, weights = self.sample(
            batch_size, self.online_net, self.target_net, beta, id)
        loss = QNet.train_model(
            self.online_net, self.target_net, self.optimizer, batch, weights)
        return loss

    def reply(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def first(self):
        self.online_net.to(device)
        self.target_net.to(device)
        self.online_net.train()
        self.target_net.train()

    def sample(self, batch_size, net, target_net, beta, id):
        probability_sum = sum(self.memory.memory_probabiliy[id])
        p = [probability /
             probability_sum for probability in self.memory.memory_probabiliy[id]]

        indexes = np.random.choice(
            np.arange(len(self.memory.memory[id])), batch_size, p=p)
        transitions = [self.memory.memory[id][idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.memory.capacity * p_j, -beta)
                   for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        weights = weights / weights.max()

        td_error = QNet.get_loss(net, target_net, batch.state,
                                 batch.next_state, batch.action, batch.reward, batch.mask)
        td_error = td_error.detach()

        td_error_idx = 0
        for idx in indexes:
            self.memory.memory_probabiliy[id][idx] = pow(
                abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            td_error_idx += 1

        return batch, weights

    def decide_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            action = torch.tensor(
                [[random.random() for _ in range(7)]])
            subaction = torch.tensor(
                [[random.uniform(0, action[0][i]) for i in range(7)]])
            action = action.view(7)
            subaction = subaction.view(7)
            return action, subaction
        else:
            return self.target_net.get_action(state)
