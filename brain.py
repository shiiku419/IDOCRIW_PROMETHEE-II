import random
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import ReplayMemory, Transition
from log import TensorboardLogger


class Brain:
    def __init__(self, num_states, num_actions, BATCH_SIZE=32, CAPACITY=10000, GAMMA=0.99):
        self.num_actions = num_actions

        self.BATCH_SIZE = BATCH_SIZE
        self.CAPACITY = CAPACITY
        self.GAMMA = GAMMA

        self.memory = ReplayMemory(CAPACITY)

        self.model = nn.Sequential()
        self.model.add_module('fc1', nn.Linear(7, 36))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('fc2', nn.Linear(36, 36))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(36, 7*10))

        self.logger = TensorboardLogger()

        # print(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def replay(self, id, episode):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        if len(self.memory.memory[id]) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE, id)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).type(torch.int64)
        reward_batch = torch.cat(batch.reward)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        self.model.eval()

        state_action_values = self.model(
            state_batch).gather(1, action_batch).max(1)[0].unsqueeze(1)

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))

        next_state_values = torch.zeros(self.BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()
            with torch.no_grad():
                out = self.model(state).view(7, 10)
                action = out.max(1)[1]  # 1,1
                subaction = out.min(1)[1]
                #action = action/10
        else:
            action = torch.tensor(
                [[random.random() for _ in range(7)]])
            action = action.view(7)
            subaction = np.zeros(7)
        return action, subaction
