import gym.spaces
import numpy as np
import matplotlib as plt
from collections import namedtuple
import random
import math
from scipy.special import softmax
import torch
import torch.nn.functional as F
from agent import Agents
from environment import Environment


class DQN:

    def __init__(self):
        self.env = Environment()
        self.obs = self.env.reset()
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agents = [Agents(i, num_states, num_actions)
                       for i in range(self.env.n_member)]

    def run2(self):

        for episode in range(20000):  # 最大試行数分繰り返す
            observation = self.env.reset()  # 環境の初期化

            state = observation  # 観測をそのまま状態sとして使用
            state = torch.from_numpy(state).type(
                torch.FloatTensor)  # NumPy変数をPyTorchのテンソルに変換

            state = torch.unsqueeze(state, 0)  # size 4をsize 1x4に変換

            for step in range(20):  # 1エピソードのループ

                # 行動a_tの実行により、s_{t+1}とdoneフラグを求める
                # actionから.item()を指定して、中身を取り出す

                for i in range(len(self.agents)):

                    action = self.agents[i].get_action(state, episode)

                    observation_next, _, done, _ = self.env.step(
                        (action/10).tolist(), i)  # rewardとinfoは使わないので_にする

                    # 報酬を与える。さらにepisodeの終了評価と、state_nextを設定する
                    if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                        state_next = None  # 次の状態はないので、Noneを格納

                        if step < 195:
                            reward = torch.FloatTensor(
                                [-1.0])  # 途中でこけたら罰則として報酬-1を与える
                            complete_episodes = 0  # 連続成功記録をリセット
                        else:
                            reward = torch.FloatTensor(
                                [1.0])  # 立ったまま終了時は報酬1を与える
                            complete_episodes = complete_episodes + 1  # 連続記録を更新
                    else:
                        reward = torch.FloatTensor([0.0])  # 普段は報酬0
                        state_next = observation_next  # 観測をそのまま状態とする
                        state_next = torch.from_numpy(state_next).type(
                            torch.FloatTensor)  # numpy変数をPyTorchのテンソルに変換
                        state_next = torch.unsqueeze(
                            state_next, 0)  # size 4をsize 1x4に変換

                    # メモリに経験を追加
                    self.agents[i].memorize(
                        state, action, state_next, reward, agent_id=i)

                    # Experience ReplayでQ関数を更新する
                    self.agents[i].update_q_function()

                    # 観測の更新
                    state = state_next

                    # 終了時の処理
                    if done:
                        print('end')
                        break

                if done:
                    break


if __name__ == "__main__":
    dqn = DQN()
    dqn.run2()
