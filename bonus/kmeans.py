import numpy as np
import random 

def _kmeans(n_clusters, X, max_iter):
    m,n = X.shape
    c_index = random.sample(range(m), n_clusters)
    centroids, labels = X[c_index], np.zeros((m,1))
    for _ in range(max_iter):
        for i in range(m):
            labels[i] = ((centroids - X[i]) ** 2).sum(axis=1).argmin()
        for k in range(n_clusters):
            r = X[np.where(labels==k)[0]][:]
            centroids[k] = r.mean(axis=0) if r.any() else np.zeros((1,n)) 
    return labels