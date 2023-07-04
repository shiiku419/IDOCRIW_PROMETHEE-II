import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import Normal
from torch.utils.data.sampler import *
import numpy as np


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0)
        elif "weight" in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor_RNN(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_RNN, self).__init__()
        self.rnn_hidden = None

        self.fc1 = nn.Linear(actor_input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.action_dim)

        # Modify this to output mean and log_std for 'thresholds' and 'matrix'
        self.fc_mean_thresholds = nn.Linear(args.rnn_hidden_dim, 5)
        self.fc_log_std_thresholds = nn.Linear(args.rnn_hidden_dim, 5)
        self.fc_mean_matrix = nn.Linear(args.rnn_hidden_dim, 5)
        self.fc_log_std_matrix = nn.Linear(args.rnn_hidden_dim, 5)

        self.activate_func = [nn.Tanh(), nn.ReLU()][args.use_relu]

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            # orthogonal_init(self.fc2, gain=0.01)
            orthogonal_init(self.fc_mean_thresholds, gain=0.01)
            orthogonal_init(self.fc_log_std_thresholds, gain=0.01)
            orthogonal_init(self.fc_mean_matrix, gain=0.01)
            orthogonal_init(self.fc_log_std_matrix, gain=0.01)

    def forward(self, actor_input):
        # When 'choose_action': actor_input.shape=(N, actor_input_dim), prob.shape=(N, action_dim)
        # When 'train':         actor_input.shape=(mini_batch_size*N, actor_input_dim),prob.shape=(mini_batch_size*N, action_dim)
        x = self.activate_func(self.fc1(actor_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        # prob = torch.softmax(self.fc2(self.rnn_hidden), dim=-1)
        # return prob

        # Output mean and log_std for 'thresholds' and 'matrix'
        mean_thresholds = self.fc_mean_thresholds(self.rnn_hidden)
        log_std_thresholds = self.fc_log_std_thresholds(self.rnn_hidden)
        mean_matrix = self.fc_mean_matrix(self.rnn_hidden)
        log_std_matrix = self.fc_log_std_matrix(self.rnn_hidden)

        return mean_thresholds, log_std_thresholds, mean_matrix, log_std_matrix


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
        critic_input = critic_input.float()
        x = self.activate_func(self.fc1(critic_input))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        value = self.fc2(self.rnn_hidden)
        return value


"""
class Actor_MLP(nn.Module):
    def __init__(self, args, actor_input_dim):
        super(Actor_MLP, self).__init__()
        self.fc1 = nn.Linear(2, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)

        # Separate output layers for 'thresholds' and 'matrix'
        # 5 outputs for 'thresholds'
        self.fc_thresholds = nn.Linear(args.mlp_hidden_dim, 5)
        # 5*5 outputs for 'matrix'
        self.fc_matrix = nn.Linear(args.mlp_hidden_dim, 5)

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
        matrix = torch.sigmoid(self.fc_matrix(x)).view(-1, 5,)

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
"""


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
        # else:
        # self.actor = Actor_MLP(args, self.actor_input_dim)
        # self.critic = Critic_MLP(args, self.critic_input_dim)

        self.ac_parameters = list(self.actor.parameters()) + list(
            self.critic.parameters()
        )
        if self.set_adam_eps:
            print("------set adam eps------")
            self.ac_optimizer = torch.optim.Adam(
                self.ac_parameters, lr=self.lr, eps=1e-5
            )
        else:
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr)

    def choose_action(self, obs_n, evaluate):
        with torch.no_grad():
            actor_inputs = []
            actor_inputs.append(obs_n)
            if self.add_agent_id:
                actor_inputs.append(torch.eye(self.N))

            actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
            actor_inputs = actor_inputs.reshape(10, 10)

            # Get means and log_stds from the actor network
            (
                mean_thresholds,
                log_std_thresholds,
                mean_matrix,
                log_std_matrix,
            ) = self.actor(actor_inputs)
            std_thresholds = log_std_thresholds.exp()
            std_matrix = log_std_matrix.exp()

            if evaluate:
                # For evaluation, choose action deterministically using means
                clipped_thresholds = np.clip(mean_thresholds.numpy(), 0, 10)
                clipped_matrix = np.clip(mean_matrix.numpy(), 0, 1)
                log_prob = None
            else:
                # For exploration, sample actions from normal distributions
                normal_thresholds = Normal(mean_thresholds, std_thresholds)
                normal_matrix = Normal(mean_matrix, std_matrix)

                # Sample actions
                sampled_thresholds = normal_thresholds.sample().numpy()
                sampled_matrix = normal_matrix.sample().numpy()

                # Clip the actions to be within the allowed range
                clipped_thresholds = np.clip(sampled_thresholds, 0, 10)
                clipped_matrix = np.clip(sampled_matrix, 0, 1)

                # Get the log probabilities of the sampled actions
                log_prob_thresholds = normal_thresholds.log_prob(
                    torch.tensor(sampled_thresholds)
                ).numpy()
                log_prob_matrix = normal_matrix.log_prob(
                    torch.tensor(sampled_matrix)
                ).numpy()
                log_prob = {
                    "thresholds": log_prob_thresholds,
                    "matrix": log_prob_matrix,
                }

            # Combine the components into the final action
            actions = []
            for i in range(self.N):
                action = {
                    "thresholds": clipped_thresholds[i],
                    "matrix": clipped_matrix[i],
                }
                actions.append(action)

            return actions, log_prob

    def get_value(self, s):
        with torch.no_grad():
            critic_inputs = []
            # Because each agent has the same global state, we need to repeat the global state 'N' times.
            s = (
                torch.tensor(s, dtype=torch.float32).unsqueeze(0).repeat(self.N, 1)
            )  # (state_dim,)-->(N,state_dim)
            critic_inputs.append(s)
            if self.add_agent_id:  # Add an one-hot vector to represent the agent_id
                critic_inputs.append(torch.eye(self.N))
            # critic_input.shape=(N, critic_input_dim)
            critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
            v_n = self.critic(critic_inputs)  # v_n.shape(N,1)
            return v_n.numpy().flatten()[: self.N]

    def train(self, replay_buffer, total_steps):
        batch = replay_buffer.get_training_data()  # get training data

        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = (
                batch["r_n"]
                + self.gamma * batch["v_n"][:, 1:] * (1 - batch["done_n"])
                - batch["v_n"][:, :-1]
            )
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch["v_n"][:, :-1]
            if self.use_adv_norm:
                adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        actor_inputs, critic_inputs = self.get_inputs(batch)

        for _ in range(self.K_epochs):
            for index in BatchSampler(
                SequentialSampler(range(self.batch_size)), self.mini_batch_size, False
            ):
                if self.use_rnn:
                    self.actor.rnn_hidden = None
                    self.critic.rnn_hidden = None
                    probs_now, values_now = [], []
                    for t in range(self.episode_limit):
                        (
                            mean_thresholds,
                            log_std_thresholds,
                            mean_matrix,
                            log_std_matrix,
                        ) = self.actor(
                            actor_inputs[index, t]
                            .reshape(self.mini_batch_size * self.N, -1)
                            .float()
                        )
                        v = self.critic(
                            critic_inputs[index, t].reshape(
                                self.mini_batch_size * self.N, -1
                            )
                        )
                        values_now.append(v.reshape(self.mini_batch_size, self.N))

                        # Take a subset of mean_thresholds and log_std_thresholds for current mini-batch
                        mini_batch_mean_thresholds = mean_thresholds.view(-1, 5)[index]
                        mini_batch_log_std_thresholds = log_std_thresholds.view(-1, 5)[
                            index
                        ]

                        # Calculate standard deviation
                        std = torch.exp(mini_batch_log_std_thresholds)

                        # Create distribution for the current mini-batch
                        dist_now = torch.distributions.Normal(
                            mini_batch_mean_thresholds, std
                        )
                        # Corrected log probability calculation
                        actions = batch["a_n"]["thresholds"].view(-1, 5)[index]
                        log_prob = dist_now.log_prob(actions)
                        probs_now.append(
                            log_prob.reshape(self.mini_batch_size, self.N, -1)
                        )

                    probs_now = torch.cat(probs_now, dim=1)
                    values_now = torch.stack(values_now, dim=1)
                else:
                    (
                        mean_thresholds,
                        log_std_thresholds,
                        mean_matrix,
                        log_std_matrix,
                    ) = self.actor(actor_inputs[index])
                    values_now = self.critic(critic_inputs[index]).squeeze(-1)

                    # Take a subset of mean_thresholds and log_std_thresholds for current mini-batch
                    mini_batch_mean_thresholds = mean_thresholds.view(-1, 5)[index]
                    mini_batch_log_std_thresholds = log_std_thresholds.view(-1, 5)[
                        index
                    ]

                    # Calculate standard deviation
                    std = torch.exp(mini_batch_log_std_thresholds)

                    # Create distribution for the current mini-batch
                    dist_now = torch.distributions.Normal(
                        mini_batch_mean_thresholds, std
                    )

                    std = torch.exp(log_std_thresholds)
                    dist_now = torch.distributions.Normal(mean_thresholds, std)
                    probs_now = dist_now.log_prob(batch["a_n"]["thresholds"][index])

                # Corrected entropy calculation
                dist_entropy = 0.5 + 0.5 * torch.log(2 * torch.tensor(np.pi) * std**2)
                a_logprob_n_now = probs_now
                batch_slice = (
                    batch["a_logprob_n"]["thresholds"][index].detach().unsqueeze(1)
                )
                ratios = torch.exp(a_logprob_n_now - batch_slice)
                ratios_chunked = ratios.view(8, 5, 25, 5)
                surr1 = ratios_chunked * adv[index].unsqueeze(1)

                # surr1 = ratios * adv[index]
                surr2 = torch.clamp(
                    ratios_chunked, 1 - self.epsilon, 1 + self.epsilon
                ) * adv[index].unsqueeze(1)
                # actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                dist_entropy_expanded = (
                    dist_entropy.unsqueeze(2).unsqueeze(3).expand_as(surr1)
                )
                actor_loss = (
                    -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy_expanded
                )

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = (
                        torch.clamp(
                            values_now - values_old, -self.epsilon, self.epsilon
                        )
                        + values_old
                        - v_target[index]
                    )
                    values_error_original = values_now - v_target[index]
                    critic_loss = torch.max(
                        values_error_clip**2, values_error_original**2
                    )
                else:
                    critic_loss = (values_now - v_target[index]) ** 2

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
            p["lr"] = lr_now

    def get_inputs(self, batch):
        actor_inputs, critic_inputs = [], []
        actor_inputs.append(batch["obs_n"]["individual"])
        critic_inputs.append(
            batch["s"]["individual"].unsqueeze(2).repeat(1, 1, self.N, 1)
        )
        if self.add_agent_id:
            # agent_id_one_hot.shape=(mini_batch_size, max_episode_len, N, N)
            agent_id_one_hot = (
                torch.eye(self.N)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(self.batch_size, self.episode_limit, 1, 1)
            )
            actor_inputs.append(agent_id_one_hot)
            critic_inputs.append(agent_id_one_hot)

        # actor_inputs.shape=(batch_size, episode_limit, N, actor_input_dim)
        actor_inputs = torch.cat([x for x in actor_inputs], dim=-1)
        # critic_inputs.shape=(batch_size, episode_limit, N, critic_input_dim)
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        actor_inputs = actor_inputs.reshape(32, 25, 5, 10)
        critic_inputs = critic_inputs.reshape(32, 25, 5, 100)
        return actor_inputs, critic_inputs

    def save_model(self, env_name, number, seed, total_steps):
        torch.save(
            self.actor.state_dict(),
            "model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
                env_name, number, seed, int(total_steps / 1000)
            ),
        )

    def load_model(self, env_name, number, seed, step):
        self.actor.load_state_dict(
            torch.load(
                "model/MAPPO_actor_env_{}_number_{}_seed_{}_step_{}k.pth".format(
                    env_name, number, seed, step
                )
            )
        )
