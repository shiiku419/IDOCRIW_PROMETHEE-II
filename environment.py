import gym.spaces
import numpy as np
import random
import math
from scipy.special import softmax


class Environment(gym.core.Env):

    def __init__(self, n_member=5):
        self.dataset = np.random.rand(5, 5) + 0.01
        self.n_member = n_member
        self.n_action = n_member
        self.action_space = gym.spaces.Discrete(self.n_action)  # actionの取りうる値
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(self.n_action,))  # 観測データの取りうる値

        self.time = 0
        self.max_step = 200*n_member

        self.P = []
        self.Q = []
        self.S = []
        self.F = []

        self.first_ranking = self.get_ranking(
            self.dataset, self.criterion_type)

        self.ranking = self.first_ranking.copy()

        self.params = {}

    def step(self, action, subaction, id):
        self.time += 1
        self.ranking = self.change_ranking(
            action, subaction, self.dataset, self.criterion_type)
        observation = self.get_observation(self.ranking)
        reward = self.reward_shaping(
            self.params, self.get_reward(self.params, id))
        done = self.check_is_done(reward)
        info = {'gsi': self.params['post_gsi']}
        return observation, reward, done, info

    def reset(self):
        self.time = 0
        self.dataset = np.random.rand(5, 5) + 0.01
        self.first_ranking = self.get_ranking(
            self.dataset, self.criterion_type)
        observation = self.get_observation(self.first_ranking)
        return observation

    def close(self):
        pass

    def seed(self):
        pass

    def get_satisfaction(self, id):
        psi, gsi = self.calc_satisfaction(
            self.distance, self.first_ranking, 1, self.n_member)

        post_psi, post_gsi = self.calc_satisfaction(
            self.distance, self.ranking, 1, self.n_member)

        self.params['pre_psi'] = psi[id]
        self.params['post_psi'] = post_psi[id]
        self.params['pre_gsi'] = gsi
        self.params['post_gsi'] = post_gsi

        return self.params

    def reward_shaping(self, params, reward):
        if params['post_psi'] - params['pre_psi'] < 0:
            return -2
        elif params['post_gsi'] - params['pre_gsi'] < 0:
            print('out', params['post_gsi'] - params['pre_gsi'])
            return -1
        else:
            return reward

    def get_reward(self, params, id):
        params = self.get_satisfaction(id)

        # pre psi
        # print(params)

        post_psi = params['post_psi']
        post_gsi = params['post_gsi']

        reward = 0

        main_reward = post_psi
        sub_reward = post_gsi

        # 補助報酬　倍率未定
        reward += main_reward + sub_reward / self.n_member

        self.first_ranking = self.ranking

        return reward

    def get_observation(self, p):
        observation = self.calc_group_rank(p)
        # print(observation)
        return observation

    def check_is_done(self, reward):
        if reward == 2:
            return True
        else:
            return self.time == self.max_step

    def idocriw_method(self, dataset, criterion_type):
        X = np.copy(dataset)
        X = X/X.sum(axis=0)
        X_ln = np.copy(dataset)
        X_r = np.copy(dataset)
        for i in range(0, X.shape[0]):
            for j in range(0, X.shape[1]):
                X_ln[i, j] = X[i, j]*math.log(X[i, j])
        d = np.zeros((1, X.shape[1]))
        w = np.zeros((1, X.shape[1]))
        for i in range(0, d.shape[1]):
            d[0, i] = 1-(-1/(math.log(d.shape[1]))*sum(X_ln[:, i]))
        for i in range(0, w.shape[1]):
            w[0, i] = d[0, i]/d.sum(axis=1)
        for i in range(0, len(criterion_type)):
            if (criterion_type[i] == 'min'):
                X_r[:, i] = dataset[:, i].min() / X_r[:, i]
        X_r = X_r/X_r.sum(axis=0)
        #a_min = X_r.min(axis = 0)
        a_max = X_r.max(axis=0)
        A = np.zeros(dataset.shape)
        np.fill_diagonal(A, a_max)
        for k in range(0, A.shape[0]):
            i, _ = np.where(X_r == a_max[k])
            i = i[0]
            for j in range(0, A.shape[1]):
                A[k, j] = X_r[i, j]
        #a_min_ = A.min(axis = 0)
        a_max_ = A.max(axis=0)
        P = np.copy(A)
        for i in range(0, P.shape[1]):
            P[:, i] = (-P[:, i] + a_max_[i])/a_max[i]
        WP = np.copy(P)
        np.fill_diagonal(WP, -P.sum(axis=0))
        # print(WP)
        return WP

    def distance_matrix(self, dataset, criteria=0):
        distance_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
        for i in range(0, distance_array.shape[0]):
            for j in range(0, distance_array.shape[1]):
                distance_array[i, j] = dataset[i,
                                               criteria] - dataset[j, criteria]
        return distance_array

    def preference_degree(self, dataset, W, Q, S, P, F):
        pd_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
        for w in range(0, dataset.shape[1]):
            W[w] = softmax(W[w], axis=0)
            for k in range(0, dataset.shape[1]):
                distance_array = self.distance_matrix(dataset, criteria=k)
                for i in range(0, distance_array.shape[0]):
                    for j in range(0, distance_array.shape[1]):
                        if (i != j):
                            if (F[k] == 't1'):
                                if (distance_array[i, j] <= 0):
                                    distance_array[i, j] = 0
                                else:
                                    distance_array[i, j] = 1
                            if (F[k] == 't2'):
                                if (distance_array[i, j] <= Q[k]):
                                    distance_array[i, j] = 0
                                else:
                                    distance_array[i, j] = 1
                            if (F[k] == 't3'):
                                if (distance_array[i, j] <= 0):
                                    distance_array[i, j] = 0
                                elif (distance_array[i, j] > 0 and distance_array[i, j] <= P[k]):
                                    distance_array[i,
                                                   j] = distance_array[i, j]/P[k]
                                else:
                                    distance_array[i, j] = 1
                            if (F[k] == 't4'):
                                if (distance_array[i, j] <= Q[k]):
                                    distance_array[i, j] = 0
                                elif (distance_array[i, j] > Q[k] and distance_array[i, j] <= P[k]):
                                    distance_array[i, j] = 0.5
                                else:
                                    distance_array[i, j] = 1
                            if (F[k] == 't5'):
                                if (distance_array[i, j] <= Q[k]):
                                    distance_array[i, j] = 0
                                elif (distance_array[i, j] > Q[k] and distance_array[i, j] <= P[k]):
                                    distance_array[i, j] = (
                                        distance_array[i, j] - Q[k])/(P[k] - Q[k])
                                else:
                                    distance_array[i, j] = 1
                            if (F[k] == 't6'):
                                if (distance_array[i, j] <= 0):
                                    distance_array[i, j] = 0
                                else:
                                    distance_array[i, j] = 1 - \
                                        math.exp(-(distance_array[i, j]
                                                   ** 2)/(2*S[k]**2))
                            if (F[k] == 't7'):
                                if (distance_array[i, j] == 0):
                                    distance_array[i, j] = 0
                                elif (distance_array[i, j] > 0 and distance_array[i, j] <= S[k]):
                                    distance_array[i, j] = (
                                        distance_array[i, j]/S[k])**0.5
                                elif (distance_array[i, j] > S[k]):
                                    distance_array[i, j] = 1
                pd_array = pd_array + softmax(W[w], axis=0)[k]*distance_array
            pd_array = pd_array/sum(W[w])
        return pd_array

    def promethee_ii(self, dataset, W, Q, S, P, F, sort=True, topn=0, graph=False):
        pd_matrix = self.preference_degree(dataset, W, Q, S, P, F)
        flow_plus = np.sum(pd_matrix, axis=1)/(pd_matrix.shape[0] - 1)
        flow_minus = np.sum(pd_matrix, axis=0)/(pd_matrix.shape[0] - 1)
        flow = flow_plus - flow_minus
        flow = np.reshape(flow, (pd_matrix.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis=1)
        if (sort == True or graph == True):
            flow = flow[np.argsort(flow[:, 1])]
            flow = flow[::-1]
        if (topn > 0):
            if (topn > pd_matrix.shape[0]):
                topn = pd_matrix.shape[0]
            # for i in range(0, topn):
                # print('alternative' + str(int(flow[i, 0])) + ': ' + str(round(flow[i, 1], 3)))
        # print(flow)
        return flow

    def distance(self, j, g_rank):
        return abs(j - g_rank)**2

    def calc_satisfaction(self, func, p, frm, to):
        result = 0
        satisfaction = 0
        group_satisfaction = 0
        satisfaction_index = []
        g_ranks = self.calc_group_rank(p)
        for i in range(0, len(p)):
            # print('DM'+str(i+1))
            i_ranks = p[i][np.argsort(p[1][:, 1])]

            for j in range(frm, to+1):
                g_rank = np.where(g_ranks == i_ranks[j-1][0])[0][0] + 1
                result += func(j, g_rank)

            bottom = to**3 - to
            satisfaction = 1 - 6 * result / bottom
            group_satisfaction += satisfaction
            satisfaction_index += [satisfaction]
        return satisfaction_index, group_satisfaction

    def calc_group_rank(self, p):
        group_rank = np.copy(p[0])
        for i in range(1, len(p)):
            group_rank += p[i]
        group_rank = group_rank/len(p)
        group_rank = group_rank[np.argsort(group_rank[:, 1])]
        return group_rank

    # Criterion Type: 'max' or 'min'
    criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']

    # Parameters

    def get_ranking(self, dataset, criterion_type):
        W = self.idocriw_method(dataset, criterion_type)
        pref = ['t1', 't2', 't3', 't4', 't5', 't6']

        p = {}

        for i in range(5):
            self.P = [random.randint(1, 10)/10 for _ in range(5)]
            self.Q = [random.uniform(0, self.P[j]) for j in range(5)]
            self.S = [(self.P[j]-self.Q[j]) for j in range(5)]
            self.F = [pref[random.randint(0, 5)] for _ in range(5)]

            p[i] = self.promethee_ii(dataset, W=W, Q=self.Q, S=self.S, P=self.P, F=self.F,
                                     sort=False, topn=10, graph=False)
        return p

    def change_ranking(self, action, subaction, dataset, criterion_type):
        W = self.idocriw_method(dataset, criterion_type)
        pref = ['t1', 't2', 't3', 't4', 't5', 't6']

        p = {}

        for i in range(5):
            P = action.view(5)/10
            Q = [random.uniform(0, P[j]) for j in range(5)]
            S = [(P[j]-Q[j]) for j in range(5)]
            F = [pref[random.randint(0, 5)] for _ in range(5)]

            p[i] = self.promethee_ii(dataset, W=W, Q=Q, S=S, P=P, F=F,
                                     sort=False, topn=10, graph=False)
        return p
