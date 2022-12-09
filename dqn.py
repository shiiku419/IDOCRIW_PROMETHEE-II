import torch
import numpy as np
import random
from environment import Environment
from agent import Agents
from log import TensorboardLogger

from utils import beta_start, initial_exploration, batch_size, update_target, goal_score, log_interval, device, replay_memory_capacity, lr


class DQN:

    def __init__(self):
        self.env = Environment()
        self.obs = self.env.reset()
        self.num_states = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n

        self.agents = [Agents(i, self.num_states, self.num_actions)
                       for i in range(self.env.n_member)]
        self.logger = TensorboardLogger()

    def main(self):
        self.env.seed(500)
        torch.manual_seed(500)

        for i in range(self.env.n_member):
            self.agents[i].update_target_model(
                self.num_states, self.num_actions, i)

            self.agents[i].train()

        episode_reward = [0 for _ in range(self.env.n_member)]
        psi = [0 for _ in range(self.env.n_member)]
        log_psi = [0 for _ in range(self.env.n_member)]
        rewards = [0 for _ in range(self.env.n_member)]
        losses = [0 for _ in range(self.env.n_member)]

        step_size = 0
        loss_step = 0
        sum_gsi = 0

        running_score = 0
        epsilon = 1.0
        steps = 0
        beta = beta_start
        loss = 0

        for episode in range(3000):
            done = False

            score = 0
            state = self.env.reset()
            state = torch.Tensor(state)
            state = state.unsqueeze(0)

            while not done:
                steps += 1

                agent = random.sample(
                    range(self.env.n_member), self.env.n_member)

                for k in range(self.env.n_member):
                    i = agent[k]
                    action, subaction = self.agents[i].get_action(
                        state, epsilon)
                    next_state, reward, done, info = self.env.step(
                        action, subaction, i)

                    next_state = torch.Tensor(next_state)
                    next_state = next_state.unsqueeze(0)

                    mask = 0 if done else 1
                    reward = reward if not done or score == 499 else -1
                    action_one_hot = np.zeros(2)
                    action_one_hot[action] = 1
                    self.agents[i].memorize(state, next_state,
                                            action_one_hot, reward, mask, id)

                    score += reward
                    state = next_state

                    episode_reward[i] += reward

                    psi[i] += info['psi']
                    log_psi[i] = info['psi']
                    sum_gsi += info['gsi']
                    rewards[i] = reward

                    if steps > initial_exploration:
                        loss = self.agents[i].train()

                        if loss != None:
                            losses[i] += loss
                            loss_step += 1

                        if steps % update_target == 0:
                            self.agents[i].update_target_model(
                                self.num_states, self.num_actions, i)

            score = score if score == 500.0 else score + 1
            running_score = 0.99 * running_score + 0.01 * score

            if done:
                self.logger.close()

            if episode % log_interval == 0:
                print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                    episode, running_score, epsilon))

                if loss_step != 0:
                    self.logger.log_value(
                        'agent/avg_loss', {'agent'+str(i): losses[i]/loss_step for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'ave_gsi', {'gsi': sum_gsi/step_size}, episode)

                self.logger.log_value(
                    'agent/reward', {'agent'+str(i): rewards[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/psi', {'agent'+str(i): log_psi[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'gsi', {'gsi': info['gsi']}, episode)

                self.logger.log_value(
                    'agent/episode_reward', {'agent'+str(i): episode_reward[i] for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/ave_reward', {'agent'+str(i): episode_reward[i]/step_size for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'agent/ave_psi', {'agent'+str(i): psi[i]/step_size for i in range(self.env.n_member)}, episode)

                self.logger.log_value(
                    'step_size', {'step_size': step_size}, episode)

                if running_score > goal_score:
                    break


if __name__ == "__main__":
    dqn = DQN()
    dqn.main()
