from brain import Brain


class Agents:
    def __init__(self, agent_id, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        self.agent_id = agent_id

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
