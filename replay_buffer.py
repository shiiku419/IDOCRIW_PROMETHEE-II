import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()
        # create a buffer (dictionary)

    def reset_buffer(self):
        self.buffer = {'obs_n': np.zeros([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.zeros([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.zeros([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_n': np.zeros([self.batch_size, self.episode_limit], dtype=object),
                       'a_logprob_n': np.zeros([self.batch_size, self.episode_limit], dtype=object),
                       'r_n': np.zeros([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.zeros([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n
        self.buffer['s'][self.episode_num][episode_step] = s
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.buffer['a_n'][self.episode_num][episode_step] = {
            'thresholds': a_n['thresholds'],
            'matrix': a_n['matrix']
        }
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = {
            'thresholds': a_logprob_n['thresholds'],
            'matrix': a_logprob_n['matrix']
        }
        self.buffer['r_n'][self.episode_num][episode_step] = r_n
        self.buffer['done_n'][self.episode_num][episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            if key == 'a_n' or key == 'a_logprob_n':
                # Assuming that self.buffer[key] is an array of dictionaries
                array_data = self.buffer[key]
                if len(array_data) > 0 and isinstance(array_data[0], dict):
                    batch[key] = [
                        {sub_key: torch.tensor(val, dtype=torch.float64)
                         for sub_key, val in item.items()}
                        for item in array_data
                    ]
                else:
                    # Handle other data formats
                    pass
            else:
                data = np.array(self.buffer[key], dtype=np.float64)
                batch[key] = torch.from_numpy(data)
        return batch
