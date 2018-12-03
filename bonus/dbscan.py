import numpy as np
def expandCluster(c, neighbors, labels, eps, min_points, distance_matrix):
    pointer = 0
    while(True):
        l_prev = len(neighbors)
        for pp in neighbors[pointer:]:
            pointer += 1
            if labels[pp] == -1:
                labels[pp] = c
            elif labels[pp] >0:
                continue
            else:
                labels[pp] = c
                r = distance_matrix[pp]
                n = list(np.where(r<eps)[0])[:]
                if len(n) >= min_points:
                    add_neighbors = set(n) - set(n).intersection(set(neighbors))
                    neighbors += list(add_neighbors)
        if len(neighbors) == l_prev:
            break
def dbscan(X, eps=1, min_points=10, verbose=False):
    m = len(X)
    distance_matrix = np.zeros((m,m))
    labels = np.zeros((m,1))
    c = 1
    for i in range(m):
        distance_matrix[i] = ((X-X[i])**2).sum(axis=1)
    for i in range(m):
        if labels[i] != 0:
            continue
        r = distance_matrix[i]
        neighbors = list(np.where(r<eps)[0])[1:]
        if len(neighbors)<min_points:
            labels[i] = -1
            continue
        c += 1
        labels[i] = c
        for p in neighbors:
            expandCluster(c,neighbors,labels,eps,min_points,distance_matrix)
    return labels