from brain import Brain


class Agents:
    def __init__(self, agent_id, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        self.agent_id = agent_id

    def update_q_function(self, id):
        self.brain.replay(id)

    def get_action(self, state, episode):
        action, subaction = self.brain.decide_action(state, episode)
        return action, subaction

    def memorize(self, state, action, state_next, reward, id):
        self.brain.memory.push(state, action, state_next, reward, id)
