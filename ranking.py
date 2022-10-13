# Required Libraries
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.8, 1.0, 0.8),))
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return

# Function: IDOCRIW
def idocriw_method(dataset, criterion_type):
    X    = np.copy(dataset)
    X    = X/X.sum(axis = 0)
    X_ln = np.copy(dataset)
    X_r  = np.copy(dataset)
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            X_ln[i,j] = X[i,j]*math.log(X[i,j])
    d    = np.zeros((1, X.shape[1]))
    w    = np.zeros((1, X.shape[1]))
    for i in range(0, d.shape[1]):
        d[0,i] = 1-( -1/(math.log(d.shape[1]))*sum(X_ln[:,i])) 
    for i in range(0, w.shape[1]):
        w[0,i] = d[0,i]/d.sum(axis = 1)
    for i in range(0, len(criterion_type)):
        if (criterion_type[i] == 'min'):
           X_r[:,i] = dataset[:,i].min() / X_r[:,i]
    X_r   = X_r/X_r.sum(axis = 0)
    #a_min = X_r.min(axis = 0)       
    a_max = X_r.max(axis = 0) 
    A     = np.zeros(dataset.shape)
    np.fill_diagonal(A, a_max)
    for k in range(0, A.shape[0]):
        i, _ = np.where(X_r == a_max[k])
        i    = i[0]
        for j in range(0, A.shape[1]):
            A[k, j] = X_r[i, j]
    #a_min_ = A.min(axis = 0)       
    a_max_ = A.max(axis = 0) 
    P      = np.copy(A)    
    for i in range(0, P.shape[1]):
        P[:,i] = (-P[:,i] + a_max_[i])/a_max[i]
    WP     = np.copy(P)
    np.fill_diagonal(WP, -P.sum(axis = 0))
    #print(WP)
    return WP
    
'''
    def target_function(variable = [0]*WP.shape[1]):
        variable = [variable[i]/sum(variable) for i in range(0, len(variable))]
        WP_s     = np.copy(WP)
        for i in range(0, WP.shape[0]):
            for j in range(0, WP.shape[1]):
                WP_s[i, j] = WP_s[i, j]*variable[j]
        total = abs(WP_s.sum(axis = 1)) 
        total = sum(total) 
        return total
    
    solution = genetic_algorithm(population_size = size, mutation_rate = 0.1, elite = 1, min_values = [0]*WP.shape[1], max_values = [1]*WP.shape[1], eta = 1, mu = 1, generations = gen, target_function = target_function)
    solution = solution[:-1]
    solution = solution/sum(solution)
    w_       = np.copy(w)
    w_       = w_*solution
    w_       = w_/w_.sum()
    w_       = w_.T
    for i in range(0, w_.shape[0]):
        print('a' + str(i+1) + ': ' + str(round(w_[i][0], 4)))
    if ( graph == True):
        flow = np.copy(w_)
        flow = np.reshape(flow, (w_.shape[0], 1))
        flow = np.insert(flow, 0, list(range(1, w_.shape[0]+1)), axis = 1)
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
        ranking(flow)
    return w_
'''

###############################################################################

# Function: Distance Matrix
def distance_matrix(dataset, criteria = 0):
    distance_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for i in range(0, distance_array.shape[0]):
        for j in range(0, distance_array.shape[1]):
            distance_array[i,j] = dataset[i, criteria] - dataset[j, criteria] 
    return distance_array

# Function: Preferences
def preference_degree(dataset, W, Q, S, P, F):
    pd_array = np.zeros(shape = (dataset.shape[0],dataset.shape[0]))
    for w in range(0, dataset.shape[1]):
        W[w] = softmax(W[w], axis=0)
        for k in range(0, dataset.shape[1]):
            distance_array = distance_matrix(dataset, criteria = k)
            for i in range(0, distance_array.shape[0]):
                for j in range(0, distance_array.shape[1]):
                    if (i != j):
                        if (F[k] == 't1'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1
                        if (F[k] == 't2'):
                            if (distance_array[i,j] <= Q[k]):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1
                        if (F[k] == 't3'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > 0 and distance_array[i,j] <= P[k]):
                                distance_array[i,j]  = distance_array[i,j]/P[k]
                            else:
                                distance_array[i,j] = 1
                        if (F[k] == 't4'):
                            if (distance_array[i,j] <= Q[k]):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                                distance_array[i,j]  = 0.5
                            else:
                                distance_array[i,j] = 1
                        if (F[k] == 't5'):
                            if (distance_array[i,j] <= Q[k]):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > Q[k] and distance_array[i,j] <= P[k]):
                                distance_array[i,j]  =  (distance_array[i,j] - Q[k])/(P[k] -  Q[k])
                            else:
                                distance_array[i,j] = 1
                        if (F[k] == 't6'):
                            if (distance_array[i,j] <= 0):
                                distance_array[i,j]  = 0
                            else:
                                distance_array[i,j] = 1 - math.exp(-(distance_array[i,j]**2)/(2*S[k]**2))
                        if (F[k] == 't7'):
                            if (distance_array[i,j] == 0):
                                distance_array[i,j]  = 0
                            elif (distance_array[i,j] > 0 and distance_array[i,j] <= S[k]):
                                distance_array[i,j]  =  (distance_array[i,j]/S[k])**0.5
                            elif (distance_array[i,j] > S[k] ):
                                distance_array[i,j] = 1
            pd_array = pd_array + softmax(W[w], axis=0)[k]*distance_array
        pd_array = pd_array/sum(W[w])
    return pd_array

# Function: Rank 
def ranking(flow):    
    rank_xy = np.zeros((flow.shape[0], 2))
    for i in range(0, rank_xy.shape[0]):
        rank_xy[i, 0] = 0
        rank_xy[i, 1] = flow.shape[0]-i           
    for i in range(0, rank_xy.shape[0]):
        if (flow[i,1] >= 0):
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (0.5, 0.8, 1.0),))
        else:
            plt.text(rank_xy[i, 0],  rank_xy[i, 1], 'a' + str(int(flow[i,0])), size = 12, ha = 'center', va = 'center', bbox = dict(boxstyle = 'round', ec = (0.0, 0.0, 0.0), fc = (1.0, 0.8, 0.8),))            
    for i in range(0, rank_xy.shape[0]-1):
        plt.arrow(rank_xy[i, 0], rank_xy[i, 1], rank_xy[i+1, 0] - rank_xy[i, 0], rank_xy[i+1, 1] - rank_xy[i, 1], head_width = 0.01, head_length = 0.2, overhang = 0.0, color = 'black', linewidth = 0.9, length_includes_head = True)
    axes = plt.gca()
    axes.set_xlim([-1, +1])
    ymin = np.amin(rank_xy[:,1])
    ymax = np.amax(rank_xy[:,1])
    if (ymin < ymax):
        axes.set_ylim([ymin, ymax])
    else:
        axes.set_ylim([ymin-1, ymax+1])
    plt.axis('off')
    plt.show() 
    return


# Function: Promethee II
def promethee_ii(dataset, W, Q, S, P, F, sort = True, topn = 0, graph = False):
    pd_matrix  = preference_degree(dataset, W, Q, S, P, F)
    flow_plus  = np.sum(pd_matrix, axis = 1)/(pd_matrix.shape[0] - 1)
    flow_minus = np.sum(pd_matrix, axis = 0)/(pd_matrix.shape[0] - 1)
    flow       = flow_plus - flow_minus 
    flow       = np.reshape(flow, (pd_matrix.shape[0], 1))
    flow       = np.insert(flow, 0, list(range(1, pd_matrix.shape[0]+1)), axis = 1)
    if (sort == True or graph == True):
        flow = flow[np.argsort(flow[:, 1])]
        flow = flow[::-1]
    if (topn > 0):
        if (topn > pd_matrix.shape[0]):
            topn = pd_matrix.shape[0]
        for i in range(0, topn):
            print('alternative' + str(int(flow[i,0])) + ': ' + str(round(flow[i,1], 3))) 
    if (graph == True):
        ranking(flow)
    return flow



# Criterion Type: 'max' or 'min'
criterion_type = ['max', 'max', 'max', 'min', 'min', 'min', 'min']

# Dataset
dataset = np.array([
                    [75.5, 420,	 74.2, 2.8,	 21.4,	0.37,	 0.16],   #a1
                    [95,   91,	 70,	 2.68, 22.1,	0.33,	 0.16],   #a2
                    [770,  1365, 189,	 7.9,	 16.9,	0.04,	 0.08],   #a3
                    [187,  1120, 210,	 7.9,	 14.4,	0.03,	 0.08],   #a4
                    [179,  875,	 112,	 4.43,	9.4,	0.016, 0.09],   #a5
                    [239,  1190, 217,	 8.51,	11.5,	0.31,	 0.07],   #a6
                    [273,  200,	 112,	 8.53,	19.9,	0.29,	 0.06]    #a7
                    ])

# PROMETHEE II

# Parameters 
Q = [ 0.3,  0.3,  0.3,  0.3,  0.3,  0.3,  0.3]
S = [ 0.4,  0.4,  0.4,  0.4,  0.4,  0.4,  0.4]
P = [ 0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
W = idocriw_method(dataset, criterion_type)
F = ['t6', 't6', 't6', 't6', 't6', 't6', 't6']

# Call Promethee II
p2 = promethee_ii(dataset, W = W, Q = Q, S = S, P = P, F = F, sort = True, topn = 10, graph = False)

import csv
f = open('out2.csv', 'w', newline='')
data = W
writer = csv.writer(f)
writer.writerows(data)
f.close()