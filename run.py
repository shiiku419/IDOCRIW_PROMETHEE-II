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

# 各種設定
num_episode = 20000  # 学習エピソード数（論文では25000になっています）
memory_size = 100000  # replay bufferの大きさ
initial_memory_size = 100000  # 最初貯める数
# ログ用の設定
episode_rewards = []
num_average_epidodes = 100

# 今回採用した環境はsimple_spreadですが、ほかにもいろいろあります
env = Environment()
max_steps = 25  # エピソードの最大ステップ数
agent = Agents()

# 最初にreplay bufferにノイズのかかった行動をしたときのデータを入れる
state = env.reset()
for step in range(initial_memory_size):
    if step % max_steps == 0:
        state = env.reset()
    actions = agent.get_action(state)
    next_state, reward, done, _ = env.step(actions)
    agent.buffer.cache(state, next_state, actions, reward, done)
    state = next_state
print('%d Data collected' % (initial_memory_size))

for episode in range(num_episode):
    state = env.reset()
    episode_reward = 0
    for t in range(max_steps):
        actions = agent.get_action(state)
        next_state, reward, done, _ = env.step(actions)
        episode_reward += sum(reward)
        agent.buffer.cache(state, next_state, actions, reward, done)
        state = next_state
        if all(done):
            break
    if episode % 4 == 0:
        agent.update()
    episode_rewards.append(episode_reward)
    if episode % 20 == 0:
        print("Episode %d finished | Episode reward %f" %
              (episode, episode_reward))

# 累積報酬の移動平均を表示
moving_average = np.convolve(episode_rewards, np.ones(
    num_average_epidodes)/num_average_epidodes, mode='valid')
plt.plot(np.arange(len(moving_average)), moving_average)
plt.title('MADDPG: average rewards in %d episodes' % num_average_epidodes)
plt.xlabel('episode')
plt.ylabel('rewards')
plt.show()

env.close()
