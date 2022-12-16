from brain import Brain


class Agents:
    def __init__(self, agent_id, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)
        self.agent_id = agent_id

    def memorize(self, state, next_state, action_one_hot, subaction_one_hot, reward, mask, id):
        self.brain.memory.push(
            state, next_state, action_one_hot, subaction_one_hot, reward, mask, id)

    def get_action(self, state, epsilon):
        action, subaction = self.brain.decide_action(
            state, epsilon)
        return action, subaction

    def update_target_model(self):
        # Target <- Net
        self.brain.reply()

    def train(self):
        self.brain.first()

    def trains(self, epsilon, beta, id):
        loss = self.brain.train(epsilon, beta, id)
        return loss
