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

        for episode in range(101):  
            observation = self.env.reset()  
            observation = np.delete(observation, 0, 1)

            state = torch.from_numpy(observation).float()

            for step in range(20):  

                for i in range(self.env.n_member):

                    action = self.agents[i].get_action(state, episode)

                    observation_next, reward, done, _ = self.env.step(
                        action, i)  

                    reward = torch.FloatTensor([reward])

                    if done:  
                        state_next = None  

                    else:
                        state_next = observation_next  
                        state_next = np.delete(state_next, 0, 1)
                        state_next = torch.from_numpy(state_next).float()

                    print(reward)
                    self.logger.log_value('reward'+str(i), reward, step)

                    self.logger.writer.flush()

                    self.agents[i].memorize(
                        state, action, state_next, reward, i)

                    self.agents[i].update_q_function(i)

                    state = state_next

                    if done:
                        break

                if done:
                    break

            print('epispde'+str(episode))

            if episode == 100:
                self.logger.close()


if __name__ == "__main__":
    dqn = DQN()
    dqn.run2()
