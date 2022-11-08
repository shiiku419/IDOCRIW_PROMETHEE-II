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


class Agents:
    def __init__(self, num_states, num_actions):
        self.discussion = Discussion(
            num_states, num_actions)

    def update_q_function(self):
        self.discussion.replay()

    def get_action(self, state, episode):
        action = self.discussion.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.discussion.memory.push(state, action, state_next, reward)
