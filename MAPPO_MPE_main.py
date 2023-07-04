import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from environment import Environment


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = Environment()  # Discrete action space
        self.args.N = 5  # The number of agents
        self.args.obs_dim_n = [{'individual': self.env.observation_space[i]['individual'].shape[0],
                                'group': self.env.observation_space[i]['group'].shape[0]}
                               for i in range(self.args.N)]  # obs dimensions of N agents
        self.args.action_dim_n = [{'thresholds': self.env.action_space[i]['thresholds'].shape[0],
                                   'matrix': self.env.action_space[i]['matrix'].shape[0]}
                                  for i in range(self.args.N)]  # actions dimensions of N agents
        # Only for homogenous agents environments like Spread in MPE,all agents have the same dimension of observation space and action space
        # The dimensions of an agent's observation space
        self.args.obs_dim = 10  # self.args.obs_dim_n[0]
        # The dimensions of an agent's action space
        self.args.action_dim = 10  # self.args.action_dim_n[0]
        # The dimensions of global state space（Sum of the dimensions of the local observation space of all agents）
        self.args.state_dim = 100  # np.sum(self.args.obs_dim_n[0])
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create N agents
        self.agent_n = MAPPO_MPE(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(
                shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(
                evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer,
                                   self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self, ):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(
            self.total_steps, evaluate_reward))
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(
            self.env_name), evaluate_reward, global_step=self.total_steps)
        # Save the rewards and models
        np.save('data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name,
                self.number, self.seed), np.array(self.evaluate_rewards))
        self.agent_n.save_model(
            self.env_name, self.number, self.seed, self.total_steps)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        obs_n = self.env.reset()
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        # If use RNN, before the beginning of each episode，reset the rnn_hidden of the Q network.
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None
        for episode_step in range(self.args.episode_limit):
            # In MPE, global state is the concatenation of all agents' local obs.
            obs_n = [np.append(item['individual'],
                               item['group']) for item in obs_n]
            obs_n = np.array(obs_n, dtype=np.float32)
            obs_n = torch.tensor(obs_n, dtype=torch.float32)
            # Get actions and the corresponding log probabilities of N agents
            a_n, a_logprob_n = self.agent_n.choose_action(
                obs_n, evaluate=evaluate)
            s = np.array(obs_n).flatten()
            # Ge,t the state values (V(s)) of N agents
            v_n = self.agent_n.get_value(s)
            obs_next_n, r_n, done_n, _ = self.env.step(a_n)
            episode_reward += r_n[0]

            if not evaluate:
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # Store the transitionpython3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
                self.replay_buffer.store_transition(
                    episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            obs_n = obs_next_n

            if done_n:
                break

        if not evaluate:
            # An episode is over, store v_n in the last step
            # obs_n = obs_n
            obs_n = [np.append(item['individual'],
                               item['group']) for item in obs_n]
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_step + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int,
                        default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25,
                        help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float,
                        default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8,
                        help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float,
                        default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float,
                        default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float,
                        default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int,
                        default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool,
                        default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool,
                        default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float,
                        default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool,
                        default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool,
                        default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool,
                        default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float,
                        default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False,
                        help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool,
                        default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float,
                        default=False, help="Whether to use value clip.")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread", number=1, seed=0)
    runner.run()
