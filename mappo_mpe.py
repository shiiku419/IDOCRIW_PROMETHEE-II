import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.utils.data.sampler import *


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # Separate output layers for thresholds and matrix
        # output shape for thresholds is (5,)
        self.fc2_thresholds = nn.Linear(args.rnn_hidden_dim, 5)
        # output shape for matrix is (5, 5)
        self.fc2_matrix = nn.Linear(args.rnn_hidden_dim, 25)

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2_thresholds, gain=0.01)
            orthogonal_init(self.fc2_matrix, gain=0.01)

    def forward(self, actor_input):
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)

        # Calculate outputs for thresholds and matrix
        output_thresholds = torch.clamp(
            self.fc2_thresholds(self.rnn_hidden), min=0, max=10)
        output_matrix = torch.clamp(self.fc2_matrix(
            self.rnn_hidden).view(-1, 5, 5), min=0, max=1)

        # Returning action as dictionary
        action = {'thresholds': output_thresholds, 'matrix': output_matrix}

        return action


class Critic_RNN(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(critic_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size*N, critic_input_dim), value.shape=(mini_batch_size*N, 1)
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(2, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)

        # Separate output layers for 'thresholds' and 'matrix'
        # 5 outputs for 'thresholds'
        self.fc_thresholds = nn.Linear(args.mlp_hidden_dim, 5)
        # 5*5 outputs for 'matrix'
        self.fc_matrix = nn.Linear(args.mlp_hidden_dim, 25)

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc_thresholds)
            orthogonal_init(self.fc_matrix, gain=0.01)

    def forward(self, actor_input):
        actor_input = actor_input.float()
        x = self.activate_func(self.fc1(actor_input))
        x = self.activate_func(self.fc2(x))

        # Applying the appropriate activation functions to limit the ranges
        thresholds = torch.sigmoid(
            self.fc_thresholds(x)) * 10  # Scaled to [0, 10]
        # Scaled to [0, 1] and reshaped to (5, 5)
        matrix = torch.sigmoid(self.fc_matrix(x)).view(-1, 5, 5)

        # Return as a dictionary
        return {'thresholds': thresholds, 'matrix': matrix}


class Critic_MLP(nn.Module):
    def __init__(self, args, critic_input_dim):
        super(Critic_MLP, self).__init__()
        self.fc1 = nn.Linear(critic_input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, critic_input_dim)
        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, critic_input):
        # When 'get_value': critic_input.shape=(N, critic_input_dim), value.shape=(N, 1)
        # When 'train':     critic_input.shape=(mini_batch_size, episode_limit, N, critic_input_dim), value.shape=(mini_batch_size, episode_limit, N, 1)
        critic_input = critic_input.float()
        x = self.activate_func(self.fc1(critic_input))
        x = self.activate_func(self.fc2(x))
        value = self.fc3(x)
        return value


class MAPPO_MPE:
    def __init__(self, args):
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = args.rnn_hidden_dim

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip

        # get the input dimension of actor and critic
        self.actor_input_dim = args.obs_dim
        self.critic_input_dim = args.state_dim
        if self.add_agent_id:
            print("------add agent id------")
            self.actor_input_dim += args.N
            self.critic_input_dim += args.N

        if self.use_rnn:
            print("------use rnn------")
            self.actor = Actor_RNN(args, self.actor_input_dim)
            self.critic = Critic_RNN(args, self.critic_input_dim)
        else:
            self.actor = Actor_MLP(args, self.actor_input_dim)
            self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(
            self.actor.parameters()) + list(self.critic.parameters())
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(
                self.ac_parameters, lr=self.lr, eps=1e-5)
        else:
            self.ac_optimizer = torch.optim.Adam(
                self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            # obs_n.shape=(N，obs_dim)
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            actor_inputs = [obs_n]

            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat(actor_inputs, dim=-1)

            # Assuming your actor network now outputs means for both 'thresholds' and 'matrix'
            # E.g., mean_outputs = self.actor(actor_inputs), where mean_outputs is a dictionary like {'thresholds': ..., 'matrix': ...}
            # this should return the dictionary {'thresholds': ..., 'matrix': ...}
            mean_outputs = self.actor(actor_inputs)

            # Define distributions for each part of the action space
            distributions = {
                'thresholds': torch.distributions.Normal(mean_outputs['thresholds'], 64),
                'matrix': torch.distributions.Normal(mean_outputs['matrix'], 64)
            }

            # Sample from the distributions
            action = {
                'thresholds': distributions['thresholds'].sample(),
                'matrix': distributions['matrix'].sample()
            }

            # Compute the log probabilities of the sampled actions
            a_logprob_n = {
                'thresholds': distributions['thresholds'].log_prob(action['thresholds']),
                'matrix': distributions['matrix'].log_prob(action['matrix'])
            }

            if evaluate:
                return action, None
            else:
                # Compute the log probabilities of the sampled actions
                a_logprob_n = {
                    'thresholds': distributions['thresholds'].log_prob(action['thresholds'].clone().detach()),
                    'matrix': distributions['matrix'].log_prob(action['matrix'].clone().detach())
                }
                return action, a_logprob_n

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = torch.tensor(s, dtype=torch.float32).unsqueeze(
                0).repeat(self.N, 1)  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            # critic_input.shape=(N, critic_input_dim)
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()[:self.N]

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and td_target have no gradient
            # deltas.shape=(batch_size,episode_limit,N)
            deltas = batch['r_n'] + self.gamma * batch['v_n'][:,
                                                              1:] * (1 - batch['done_n']) - batch['v_n'][:, : -1]
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            # adv.shape(batch_size,episode_limit,N)
            adv = torch.stack(adv, dim=1)
            # v_target.shape(batch_size,episode_limit,N)
            v_target = adv + batch['v_n'][:, : -1]
            if self.use_adv_norm:  # Trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        actor_inputs, critic_inputs = self.get_inputs(batch)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):

                # Assuming that actor network outputs means and std devs for the normal distribution
                output = self.actor(actor_inputs[index])
                means_now = output['thresholds']
                std_devs_now = output['matrix']

                dist_thresholds = torch.distributions.Normal(
                    means_now['thresholds'], std_devs_now['thresholds'])
                dist_matrix = torch.distributions.Normal(
                    means_now['matrix'], std_devs_now['matrix'])

                # Use dist_thresholds and dist_matrix for action-related computations.
                action_thresholds_logprobs = dist_thresholds.log_prob(
                    batch['a_n'][index]['thresholds'])
                action_matrix_logprobs = dist_matrix.log_prob(
                    batch['a_n'][index]['matrix'])

                # Assuming that you want to combine these logprobs in some way,
                # for example, by summing them for simplicity.
                # Note: The way you combine them depends on your specific use case.
                combined_action_logprobs = action_thresholds_logprobs + action_matrix_logprobs

                # The rest of the code should then use combined_action_logprobs in place
                # of the old a_logprob_n_now for further computations.
                ratios = torch.exp(combined_action_logprobs -
                                   batch['a_logprob_n'][index].detach())
                # ... continue with the rest of the computations as before, but using the updated ratios ...

                # Compute surrogate loss
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon,
                                    1 + self.epsilon) * adv[index]

                # Compute actor loss considering also the entropy for exploration
                actor_loss = -torch.min(surr1, surr2) + \
                    self.entropy_coef * dist_now.entropy().mean()

                # Assuming that critic network outputs values for the critic
                values_now = self.critic(critic_inputs[index]).squeeze(-1)

                # Compute critic loss
                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(
                        values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(
                        values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

                # Perform optimization step
                self.ac_optimizer.zero_grad()
                ac_loss = actor_loss.mean() + critic_loss.mean()
                ac_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                self.ac_optimizer.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch['obs_n'])
        critic_inputs.append(batch['s'].unsqueeze(2).repeat(1, 1, self.N, 1))
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = torch.eye(self.N).unsqueeze(0).unsqueeze(
                0).repeat(self.batch_size, self.episode_limit, 1, 1)
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(self.actor.state_dict(), "model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
            env_name, number, seed, int(total_steps / 1000)))

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(torch.load(
            "model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(env_name, number, seed, step)))
