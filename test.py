# Required Libraries
import csv
import random
import math
import numpy as np
from scipy.special import softmax
from collections import deque
from pyDecision.util.ga import genetic_algorithm

# Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']
# Dataset
dataset = np.array([
    [75.5, 420,	 200,    2.8,	21.4,	0.37,	 0.16],  # a1
    [95,   900,	 170,	 2.68,  22.1,	0.33,	 0.16],  # a2
    [770,  1365, 189,	 7.9,	16.9,	0.04,	 0.08],  # a3
    [187,  1120, 210,	 7.9,	14.4,	0.03,	 0.08],  # a4
    [179,  875,	 112,	 4.43,	9.4,	0.41,    0.09],  # a5
    [239,  1190, 217,	 8.51,	11.5,	0.31,	 0.07],  # a6
    [273,  1200, 112,	 8.53,	19.9,	0.29,	 0.06],  # a7
])


def idocriw_method(dataset, criterion_type):
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
    # a_min = X_r.min(axis = 0)
    a_max = X_r.max(axis=0)
    A = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)
    for k in range(0, A.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]
    # a_min_ = A.min(axis = 0)
    a_max_ = A.max(axis=0)
    P = np.copy(A)
    for i in range(0, P.shape[1]):
        P[:, i] = (-P[:, i] + a_max_[i])/a_max[i]
    WP = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis=0))
    return WP, w


WP, w = idocriw_method(dataset, criterion_type)


def target_function(variable):
    variable = [variable[i]/sum(variable) for i in range(0, len(variable))]
    WP_s = np.copy(WP)
    for i in range(0, WP.shape[0]):
        for j in range(0, WP.shape[1]):
            WP_s[i, j] = WP_s[i, j]*variable[j]
    total = abs(WP_s.sum(axis=1))
    total = sum(total)
    return total


def solution():
    solution = genetic_algorithm(population_size=5, mutation_rate=0.1, elite=1, min_values=[
                                 0]*WP.shape[1], max_values=[1]*WP.shape[1], eta=1, mu=1, generations=100, target_function=target_function)
    solution = solution[:-1]
    solution = solution/sum(solution)
    w_ = np.copy(w)
    w_ = w_*solution
    w_ = w_/w_.sum()
    w_ = w_.T

    result = [item for i in w_ for item in i]
    print(result)

    return result


def distance_matrix(dataset, criteria=0):
    distance_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i, j] = dataset[i, criteria] - dataset[j, criteria]
    return distance_array


def preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros(shape=(dataset.shape[0], dataset.shape[0]))
    for k in range(0, dataset.shape[1]):
        distance_array = distance_matrix(dataset, criteria=k)
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
        pd_array = pd_array + softmax(W, axis=0)[k]*distance_array
    return pd_array


def promethee_ii(dataset, W, Q, S, P, F, sort=True, topn=0, graph=False):
    pd_matrix = preference_degree(dataset, W, Q, S, P, F)
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


def distance(j, g_rank):
    return abs(j - g_rank)**2


def calc_satisfaction(func, p, frm, to, agent):
    result = 0
    satisfaction = 0
    satisfaction_index = [0 for _ in range(5)]
    g_ranks = calc_group_rank(p)
    for k in range(0, len(p)):
        i = agent[k]
        print(i)
        print('DM'+str(i+1))
        i_ranks = p[i][np.argsort(p[1][:, 1])]
        for j in range(frm, to+1):
            g_rank = np.where(g_ranks == i_ranks[j-1][0])[0][0] + 1
            result += func(j, g_rank)
        bottom = to**3 - to
        satisfaction = 1 - 6 * result / bottom
        satisfaction_index[i] = satisfaction
    print(satisfaction_index)
    return satisfaction_index


def calc_group_rank(p):
    group_rank = np.copy(p[0])
    for i in range(1, len(p)):
        group_rank += p[i]
    group_rank = group_rank/len(p)
    group_rank = group_rank[np.argsort(group_rank[:, 1])]
    return group_rank


# Parameters
W = solution()
pref = ['t1', 't2', 't3', 't4', 't5', 't6']
p = {}
agent = random.sample(range(5), 5)
for k in range(5):
    i = agent[k]
    P = [random.randint(1, 10)/10 for _ in range(7)]
    Q = [random.uniform(0, P[j]) for j in range(7)]
    S = [(P[j]-Q[j]) for j in range(7)]
    F = [pref[random.randint(0, 5)] for _ in range(7)]
    p[i] = promethee_ii(dataset, W=W, Q=Q, S=S, P=P, F=F,
                        sort=False, topn=10, graph=False)
index = calc_satisfaction(distance, p, 1, 7, agent)
