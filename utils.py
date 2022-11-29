from collections import namedtuple
import random
from collections import deque

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = [[] for _ in range(5)]
        self.index = 0

    def push(self, state, action, state_next, reward, id):

        if len(self.memory[id]) < self.capacity:
            self.memory[id].append(None)

        action = action.view(1, 5)
        self.memory[id][self.index] = Transition(
            state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, id):
        return random.sample(self.memory[id], batch_size)

    def __len__(self):
        return len(self.memory)
