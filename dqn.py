import torch
import numpy as np
import random
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

    def run2(self):

        for episode in range(30001):
            observation = self.env.reset()
            observation = np.delete(observation, 0, 1)

            state = torch.from_numpy(
                observation).float().view(1, self.env.n_member)

            episode_reward = [0 for _ in range(self.env.n_member)]

            step_size = 0

            for step in range(20):

                step_size += 1

                rewards = {}
                psi = {}

                agent = random.sample(
                    range(self.env.n_member), self.env.n_member)

                for k in range(self.env.n_member):

                    i = agent[k]

                    action, subaction = self.agents[i].get_action(
                        state, episode)

                    observation_next, reward, done, info = self.env.step(
                        action, subaction, i)

                    episode_reward[i] += reward

                    reward = torch.FloatTensor([reward])

                    rewards[i] = reward

                    psi[i] = info['psi']

                    self.logger.log_value(
                        'gsi', {'gsi': info['gsi']}, episode)

                    if done & i == self.env.n_member:
                        state_next = None
                        break

                    else:
                        state_next = observation_next
                        state_next = np.delete(state_next, 0, 1)
                        state_next = torch.from_numpy(
                            state_next).float().view(1, self.env.n_member)

                    self.agents[i].memorize(
                        state, action.view(1, self.env.n_member), state_next, reward, i)

                    self.agents[i].update_q_function(i, episode)

                    state = state_next

                # 意見の創発
                if step == 10:
                    self.env.generate()

                self.logger.log_value(
                    'agent/reward', {'agent'+str(i): rewards[i] for i in range(self.env.n_member)}, episode)

                self.logger.writer.flush()

                if done:
                    break

            print('epispde'+str(episode))

            self.logger.log_value(
                'agent/step_reward', {'agent'+str(i): episode_reward[i] for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'agent/psi', {'agent'+str(i): psi[i] for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'step_size', {'step_size': step_size}, episode)

            if episode < 30000:
                self.logger.close()


if __name__ == "__main__":
    dqn = DQN()
    dqn.run2()
