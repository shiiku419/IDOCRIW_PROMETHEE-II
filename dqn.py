import torch
import numpy as np
from agent import Agents
from environment import Environment
from log import TensorboardLogger


class DQN:

    def __init__(self):
        self.env = Environment()
        self.obs = self.env.reset()
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agents = [Agents(i, num_states, num_actions)
                       for i in range(self.env.n_member)]
        self.logger = TensorboardLogger()
        self.iter_no = 0

    def run2(self):

        for episode in range(100):  # 最大試行数分繰り返す
            observation = self.env.reset()  # 環境の初期化
            observation = np.delete(observation, 0, 1)

            state = torch.from_numpy(observation).float()

            for step in range(20):  # 1エピソードのループ

                for i in range(self.env.n_member):

                    action = self.agents[i].get_action(state, episode)

                    observation_next, reward, done, _ = self.env.step(
                        action, i)  

                    reward = torch.FloatTensor([reward])

                    if done:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる
                        state_next = None  # 次の状態はないので、Noneを格納

                    else:
                        state_next = observation_next  # 観測をそのまま状態とする
                        state_next = np.delete(state_next, 0, 1)
                        state_next = torch.from_numpy(state_next).float()

                    self.logger.log_value('state'+str(i), state.squeeze().ndim, step)
                    self.logger.log_value('reward'+str(i), reward, step)

                    self.logger.writer.flush()

                    # メモリに経験を追加
                    self.agents[i].memorize(
                        state, action, state_next, reward, i)

                    # Experience ReplayでQ関数を更新する
                    self.agents[i].update_q_function(i)

                    # 観測の更新
                    state = state_next

                    # 終了時の処理
                    if done:
                        break

                if done:
                    break

            print('epispde'+str(episode))

            if episode == 100:
                TensorboardLogger.close()


if __name__ == "__main__":
    dqn = DQN()
    dqn.run2()
