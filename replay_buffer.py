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
                       'a_n': [[{} for _ in range(self.episode_limit)] for _ in range(self.batch_size)],
                       'a_logprob_n': [[{} for _ in range(self.episode_limit)] for _ in range(self.batch_size)],
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
            if key == 'a_n':
                t_data_list = [torch.from_numpy(self.buffer[key][i][j]['thresholds'])
                               for i in range(self.batch_size) for j in range(self.episode_limit) if self.buffer[key][i][j]]
                m_data_list = [torch.from_numpy(self.buffer[key][i][j]['matrix'])
                               for i in range(self.batch_size) for j in range(self.episode_limit) if self.buffer[key][i][j]]

                t_data_stacked = torch.stack(t_data_list, dim=0)
                m_data_stacked = torch.stack(m_data_list, dim=0)

                batch[key] = {}
                batch[key]['thresholds'] = t_data_stacked
                batch[key]['matrix'] = m_data_stacked
            elif key == 'a_logprob_n':
                t_data_list = [torch.tensor(self.buffer[key][i][j]['thresholds'])
                               for i in range(self.batch_size) for j in range(self.episode_limit) if self.buffer[key][i][j]]
                m_data_list = [torch.tensor(self.buffer[key][i][j]['matrix'])
                               for i in range(self.batch_size) for j in range(self.episode_limit) if self.buffer[key][i][j]]

                t_data_stacked = torch.stack(t_data_list, dim=0)
                m_data_stacked = torch.stack(m_data_list, dim=0)

                batch[key] = {}
                batch[key]['thresholds'] = t_data_stacked
                batch[key]['matrix'] = m_data_stacked
            else:
                data = np.array(self.buffer[key], dtype=np.float64)
                batch[key] = torch.from_numpy(data)
        return batch
