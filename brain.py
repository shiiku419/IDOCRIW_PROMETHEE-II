import random
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from utils import ReplayMemory, Transition


class Brain:
    def __init__(self, num_states, num_actions, BATCH_SIZE=32, CAPACITY=10000, GAMMA=0.99):
        self.num_actions = num_actions  # CartPoleの行動（右に左に押す）の2を取得

        self.BATCH_SIZE = BATCH_SIZE
        self.CAPACITY = CAPACITY
        self.GAMMA = GAMMA

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory(CAPACITY)

        # ニューラルネットワークを構築
        self.model = nn.Sequential()
        #self.model.add_module('fc1', nn.Linear(num_states, 1))
        self.model.add_module('relu1', nn.ReLU())
        #self.model.add_module('fc2', nn.Linear(1, 1))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(2, num_actions))

        # print(self.model)  # ネットワークの形を出力

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def replay(self, agent_id):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        print(agent_id)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(torch.tensor(batch.action))
        reward_batch = torch.cat(batch.reward)
        print(state_batch.shape)
        print(len(batch.action))
        print(action_batch.shape)
        print(reward_batch.shape)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        self.model.eval()

        state_action_values = self.model(state_batch).gather(
            1, action_batch.type(torch.int64))

        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)))

        next_state_values = torch.zeros(self.BATCH_SIZE)

        next_state_values[non_final_mask] = self.model(
            non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + self.GAMMA * next_state_values

        self.model.train()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values.unsqueeze(1))

        # 4.3 結合パラメータを更新する
        self.optimizer.zero_grad()  # 勾配をリセット
        loss.backward()  # バックプロパゲーションを計算
        self.optimizer.step()  # 結合パラメータを更新

    def decide_action(self, state, episode):
        '''現在の状態に応じて、行動を決定する'''
        # ε-greedy法で徐々に最適行動のみを採用する
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(7, 1)  # 1,1
        else:
            # 0,1の行動をランダムに返す
            action = torch.tensor(
                [[random.random() for _ in range(self.num_actions)]])  # 0,1の行動をランダムに返す
            # actionは[torch.LongTensor of size 1x1]の形になります
            action = action.view(7, 1)
        return action
