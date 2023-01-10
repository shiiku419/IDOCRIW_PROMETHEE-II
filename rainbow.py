import torch
import numpy as np
import random
from environment import Environment
from agent import Agents
from log import TensorboardLogger

from utils import beta_start, initial_exploration, update_target, goal_score, log_interval


class Rainbow:

    def __init__(self):
        self.env = Environment()
        self.obs = self.env.reset()
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agents = [Agents(i, self.num_states, self.num_actions)
                       for i in range(self.env.n_member)]
        self.logger = TensorboardLogger()

    def main(self):
        self.env.seed()
        torch.manual_seed(500)

        for i in range(self.env.n_member):
            self.agents[i].update_target_model()

            self.agents[i].train()

        running_score = 0
        epsilon = 1.0
        steps = 0
        beta = beta_start
        loss = 0

        for episode in range(100000):
            done = False

            score = 0
            state = self.env.reset()
            state = torch.Tensor(state)

            substate = self.env.dataset
            substate = torch.Tensor(substate)

            state = np.delete(state, 0, 1).view(1, 7)
            substate = substate[0].view(1, 7)

            episode_reward = [0 for _ in range(self.env.n_member)]
            psi = [0 for _ in range(self.env.n_member)]
            log_psi = [0 for _ in range(self.env.n_member)]
            rewards = [0 for _ in range(self.env.n_member)]
            losses = [0 for _ in range(self.env.n_member)]
            gap = [0 for _ in range(self.env.n_member)]

            discuss = 0
            step = 0
            loss_step = 0
            sum_gsi = 0

            while not done:

                steps += 1
                step += 1

                agent = random.sample(
                    range(self.env.n_member), self.env.n_member)

                for k in range(self.env.n_member):
                    discuss += 1

                    i = agent[k]
                    action, subaction = self.agents[i].get_action(
                        state, substate, epsilon)

                    next_state, next_substate, reward, done, info = self.env.step(
                        action, subaction, i)

                    next_state = torch.Tensor(next_state)
                    next_state = np.delete(next_state, 0, 1).view(1, 7)

                    next_substate = torch.Tensor(next_substate)
                    next_substate = next_substate[0].view(1, 7)

                    mask = 0 if done else 1
                    action_one_hot = np.zeros(7)
                    subaction_one_hot = np.zeros(7)
                    reward = reward if not done or score == 499 else -1

                    action_one_hot[torch.argmax(action)] = 1
                    subaction_one_hot[torch.argmax(subaction)] = 1
                    self.agents[i].memorize(state, substate, next_state, next_substate,
                                            action_one_hot, subaction_one_hot, reward, mask, i)

                    score += reward
                    state = next_state

                    episode_reward[i] += reward

                    psi[i] += info['psi']
                    log_psi[i] = info['psi']
                    sum_gsi += info['gsi']
                    rewards[i] = reward
                    gap[i] = info['gap']

                    if done:
                        break

                    if steps > initial_exploration:
                        epsilon -= 0.00005
                        epsilon = max(epsilon, 0.1)
                        beta += 0.00005
                        beta = min(1, beta)

                        loss = self.agents[i].trains(
                            epsilon, beta, i)

                        if loss != None:
                            losses[i] += loss
                            loss_step += 1

                        if steps % update_target == 0:
                            self.agents[i].update_target_model()

                # 意見の創発
                if discuss % 35 == 0:
                    self.env.generate(subaction)

                if done:
                    break

            score = score if score == 500.0 else score + 1
            running_score = 0.99 * running_score + 0.01 * score

            if episode % log_interval == 0:
                print('{} episode | score: {:.2f} | epsilon: {:.2f} | step: {:.2f} | reward {:.2f} | gsi: {:.2f}'.format(
                    episode, running_score, epsilon, discuss, reward, info['gsi']))

                if loss_step != 0:
                    self.logger.log_value(
                        'agent/avg_loss', {'agent'+str(i): losses[i]/loss_step for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/reward', {'agent'+str(i): rewards[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/ave_reward', {'agent'+str(i): rewards[i]/step for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/ave_gap', {'agent'+str(i): gap[i]/step for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/psi', {'agent'+str(i): log_psi[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/episode_reward', {'agent'+str(i): episode_reward[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'log/gsi', {'gsi': info['gsi']}, episode)

                self.logger.log_value(
                    'log/discuss_length', {'discuss_length': discuss}, episode)

                if running_score > goal_score:
                    self.logger.close()
                    break


if __name__ == "__main__":
    run = Rainbow()
    run.main()
