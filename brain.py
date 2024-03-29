import numpy as np
import random
import torch.nn as nn
import torch
import torch.optim as optim
from collections import namedtuple
from model import QNet
from utils import Memory, batch_size, device, small_epsilon, alpha, replay_memory_capacity, lr
from environment import Environment

Transition = namedtuple(
    'Transition', ('state', 'substate', 'next_state', 'next_substate', 'action', 'subaction', 'reward', 'mask'))


class Brain:

    def __init__(self, num_inputs, num_actions):
        self.num_actions = num_actions

        self.memory = Memory(replay_memory_capacity)
        self.env = Environment()

        self.online_net = QNet(num_inputs, num_actions)
        self.target_net = QNet(num_inputs, num_actions)

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def train(self, epsilon, beta, id):

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

        if np.isnan(probability_sum):
            probability_sum = sum(np.nan_to_num(
                self.memory.memory_probabiliy[id], nan=0.0001))

            p = [probability /
                 probability_sum for probability in np.nan_to_num(self.memory.memory_probabiliy[id], nan=0.0001)]

        indexes = np.random.choice(
            np.arange(len(self.memory.memory[id])), batch_size, p=p)
        transitions = [self.memory.memory[id][idx] for idx in indexes]
        transitions_p = [p[idx] for idx in indexes]
        batch = Transition(*zip(*transitions))

        weights = [pow(self.memory.capacity * p_j, -beta)
                   for p_j in transitions_p]
        weights = torch.Tensor(weights).to(device)
        weights = weights / weights.max()

        td_error = QNet.get_loss(net, target_net, batch.state, batch.substate,
                                 batch.next_state, batch.next_substate, batch.action, batch.subaction, batch.reward, batch.mask)
        td_error = td_error.detach()

        td_error_idx = 0
        for idx in indexes:
            self.memory.memory_probabiliy[id][idx] = pow(
                abs(td_error[td_error_idx]) + small_epsilon, alpha).item()
            td_error_idx += 1

        return batch, weights

    def decide_action(self, state, substate, epsilon):
        if np.random.rand() <= epsilon:
            action = torch.tensor(
                [[random.randint(0, 10)/10 for _ in range(7)]])
            substate[0] = substate[0]*np.random.normal(1, 0.2, 1)
            action = action.view(7)
            subaction = substate[0].view(7)
            return action, subaction
        else:
            return self.target_net.get_action(state, substate)
