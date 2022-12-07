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

    def run(self):

        for episode in range(50001):
            observation = self.env.reset()
            observation = np.delete(observation, 0, 1)

            state = torch.from_numpy(
                observation).float().view(1, 7)

            episode_reward = [0 for _ in range(self.env.n_member)]
            psi = [0 for _ in range(self.env.n_member)]
            log_psi = [0 for _ in range(self.env.n_member)]
            losses = [0 for _ in range(self.env.n_member)]

            step_size = 0
            loss_step = 0
            sum_gsi = 0

            for step in range(50):

                step_size += 1

                agent = random.sample(
                    range(self.env.n_member), self.env.n_member)

                for k in range(self.env.n_member):

                    i = agent[k]

                    action, subaction = self.agents[i].get_action(
                        state, episode)

                    observation_next, reward, done, info = self.env.step(
                        action, subaction, i)

                    episode_reward[i] += reward

                    psi[i] += info['psi']
                    log_psi[i] = info['psi']
                    sum_gsi += info['gsi']

                    reward = torch.FloatTensor([reward])

                    self.logger.log_value(
                        'gsi', {'gsi': info['gsi']}, episode)

                    if done & k == self.env.n_member:
                        state_next = None
                        break

                    else:
                        state_next = observation_next
                        state_next = np.delete(state_next, 0, 1)
                        state_next = torch.from_numpy(
                            state_next).float().view(1, 7)

                    self.agents[i].memorize(
                        state, action.view(1, 7), state_next, reward, i)

                    loss = self.agents[i].update_q_function(i, episode)

                    if loss != None:
                        losses[i] += loss
                        loss_step += 1

                    state = state_next

                # 意見の創発
                if step % 20 == 0:
                    self.env.generate()

                if done:
                    break

                self.logger.log_value(
                    'agent/reward', {'agent'+str(i): episode_reward[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/psi', {'agent'+str(i): log_psi[i] for i in range(self.env.n_member)}, episode)

            print('epispde'+str(episode))

            if loss_step != 0:
                self.logger.log_value(
                    'agent/avg_loss', {'agent'+str(i): losses[i]/loss_step for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'ave_gsi', {'gsi': sum_gsi/step_size}, episode)

            self.logger.log_value(
                'agent/episode_reward', {'agent'+str(i): episode_reward[i] for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'agent/ave_reward', {'agent'+str(i): episode_reward[i]/step_size for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'agent/ave_psi', {'agent'+str(i): psi[i]/step_size for i in range(self.env.n_member)}, episode)

            self.logger.log_value(
                'step_size', {'step_size': step_size}, episode)

            self.logger.writer.flush()

            if episode < 50000:
                self.logger.close()


if __name__ == "__main__":
    dqn = DQN()
    dqn.run()
