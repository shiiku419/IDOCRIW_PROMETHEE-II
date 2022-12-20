import gym.spaces
import numpy as np
import random
import math
from scipy.special import softmax
import csv

f = open('action.csv', 'w')
writer = csv.writer(f)


class Environment(gym.core.Env):

    def __init__(self, n_member=5):
        self.dataset = np.random.rand(7, 7)
        self.n_member = n_member
        self.n_action = 7
        self.action_space = gym.spaces.Discrete(self.n_action)
        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(self.n_action,))

        self.time = 0
        self.max_step = 100
        self.agent = random.sample(range(self.n_member), self.n_member)
        # TODO問題空間の変数を作って大きくしていく

        self.W = None
        self.F = {}
        self.criterion_type = self.set_criterion()

        self.first_ranking = self.get_ranking(
            self.F, self.dataset, self.criterion_type)

        self.ranking = self.first_ranking.copy()

        self.pre_threshold = 0

        self.params = {}

    def step(self, action, subaction, id):
        self.time += 1
        self.ranking, penalty = self.change_ranking(
            action, subaction, id, self.dataset, self.criterion_type, self.ranking)
        observation = self.get_observation(self.ranking)
        reward, post_psi = self.get_reward(penalty, self.params, id)
        done = self.check_is_done(post_psi)
        info = {'gsi': self.params['post_gsi'],
                'psi': self.params['post_psi'],
                'gap': penalty}
        if done:
            writer.writerow([id, '+', action])
            writer.writerow([id, '-', subaction])
        return observation, reward, done, info

    def generate(self):
        random = np.random.randint(0, 4)
        index = np.where(self.ranking[random] ==
                         self.ranking[random].max(0)[1])[0][0]
        self.dataset[index] = self.dataset[index]*np.random.normal(1, 0.2, 1)

    def reset(self):
        self.time = 0
        self.criterion_type = self.set_criterion()
        self.agent = random.sample(range(self.n_member), self.n_member)
        self.dataset = np.random.rand(7, 7)
        self.first_ranking = self.get_ranking(
            self.F, self.dataset, self.criterion_type)
        observation = self.get_observation(self.first_ranking)
        return observation

    def close(self):
        pass

    def seed(self):
        pass

    def set_criterion(self):
        type = ['max', 'min']
        prob = [0.7, 0.3]
        self.criterion_type = np.random.choice(a=type, size=7, p=prob)
        return self.criterion_type

    def get_satisfaction(self, id):
        psi, gsi = self.calc_satisfaction(
            self.distance, self.first_ranking, 1, 7)

        post_psi, post_gsi = self.calc_satisfaction(
            self.distance, self.ranking, 1, 7)

        self.params['pre_psi'] = psi[id]
        self.params['post_psi'] = post_psi[id]
        self.params['pre_gsi'] = gsi
        self.params['post_gsi'] = post_gsi

        return self.params, post_psi

    def get_reward(self, penalty, params, id):
        params, post_psi = self.get_satisfaction(id)

        reward = 0
        clip = 0

        main_reward = params['post_psi'] - params['pre_psi']
        sub_reward = params['post_gsi'] - params['pre_gsi']

        clip += main_reward + (sub_reward / self.n_member)*random.random()

        self.first_ranking = self.ranking

        if clip > 0:
            reward = 1
        elif clip < 0:
            reward = -1
        else:
            reward = 0.5

        if penalty < 0:
            reward += -5
        elif penalty == 0:
            reward += -1
        else:
            reward += 5

        return reward, post_psi

    def get_observation(self, p):
        observation = self.calc_group_rank(p)
        return observation

    def check_is_done(self, post_psi):
        if all(0.9 <= flag for flag in post_psi) == True:
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
        a_max = X_r.max(axis=0)
        A = np.zeros(dataset.shape)
        np.fill_diagonal(A, a_max)
        for k in range(0, A.shape[0]):
            i, _ = np.where(X_r == a_max[k])
            i = i[0]
            for j in range(0, A.shape[1]):
                A[k, j] = X_r[i, j]
        a_max_ = A.max(axis=0)
        P = np.copy(A)
        for i in range(0, P.shape[1]):
            P[:, i] = (-P[:, i] + a_max_[i])/a_max[i]
        WP = np.copy(P)
        np.fill_diagonal(WP, -P.sum(axis=0))
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
        return flow

    def distance(self, j, g_rank):
        return abs(j - g_rank)**2

    def calc_satisfaction(self, func, p, frm, to):
        result = 0
        satisfaction = 0
        group_satisfaction = 0
        satisfaction_index = [0 for _ in range(self.n_member)]
        g_ranks = self.calc_group_rank(p)
        for k in range(0, len(p)):
            i = self.agent[k]
            i_ranks = p[i][np.argsort(p[1][:, 1])]

            for j in range(frm, to+1):
                g_rank = np.where(g_ranks == i_ranks[j-1][0])[0][0] + 1
                result += func(j, g_rank)

            bottom = to**3 - to
            satisfaction = 1 - 6 * result / bottom
            group_satisfaction += satisfaction
            satisfaction_index[i] = satisfaction
        return satisfaction_index, group_satisfaction

    def calc_group_rank(self, p):
        group_rank = np.copy(p[0])
        for i in range(1, len(p)):
            group_rank += p[i]
        group_rank = group_rank/len(p)
        group_rank = group_rank[np.argsort(group_rank[:, 1])]
        return group_rank

    def get_ranking(self, F, dataset, criterion_type):
        self.W = [self.idocriw_method(dataset, criterion_type)]*self.n_member
        pref = ['t1', 't2', 't3', 't4', 't5', 't6']

        p = {}

        for k in range(self.n_member):
            i = self.agent[k]

            self.W[i] = self.W[i]*random.random()

            P = [random.random() for _ in range(7)]
            Q = [random.uniform(0, P[j])for j in range(7)]
            S = [(P[j]-Q[j]) for j in range(7)]

            F[i] = [pref[random.randint(0, 5)] for _ in range(7)]

            self.pre_threshold = sum(S)

            p[i] = self.promethee_ii(dataset, W=self.W[i], Q=Q, S=S, P=P, F=F[i],
                                     sort=False, topn=10, graph=False)
        return p

    def change_ranking(self, action, subaction, id, dataset, criterion_type, ranking):

        P = action.view(7)/10
        Q = subaction.view(7)/10
        S = [(P[j]-Q[j]) for j in range(7)]

        penalty = sum(S) - self.pre_threshold

        ranking[id] = self.promethee_ii(dataset, W=self.W[id], Q=Q, S=S, P=P, F=self.F[id],
                                        sort=False, topn=10, graph=False)
        return ranking, penalty
