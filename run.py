import gym.spaces
import numpy as np
import matplotlib as plt
from collections import namedtuple
import random
import math
from scipy.special import softmax
import torch.nn.functional as F
from agent import Agents
from environment import Environment

env = Environment()

obs = env.reset()

for _ in range(200):
    action = env.action_space.sample()
    obs, re, done, info = env.step(action)
    if done:
        env.reset()
