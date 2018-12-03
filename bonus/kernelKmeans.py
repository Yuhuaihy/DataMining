import numpy as np
import random
def GaussianKernel(X):
    n = len(X)
    sigma = 2
    kernel_matrix = np.zeros((n,n))
    for i in range(n):
        kernel_matrix[i] = ((X-X[i]) ** 2).sum(axis=1).reshape((1,n))
    kernel_matrix = np.exp((-1/(2*(sigma**2)))* kernel_matrix)
    return kernel_matrix

def kernel_km(X,n_clusters=3, verbose=False):
    kernel_matrix = GaussianKernel(X)
    m = len(X)
    centroids = random.sample(range(m),n_clusters)
    labels = np.zeros((m,1))
    labels = labels - 1
    ind = 1
    for x in centroids:
        labels[x] = ind
        ind += 1
    distance = np.zeros((n_clusters,1))
    for _ in range(100):
        for i in range(m):
            for k in range(n_clusters):
                points = np.where(labels==k)[0]
                num = len(points)
                addend1 = kernel_matrix[i][i] ** 2
                r1 = kernel_matrix[i][:]
                r2 =r1[points]
                addend2 = -2 * r2.sum() / num
                r3 = kernel_matrix[points]
                r3 = r3[:,points]
                addend3 = r3.sum() / (num**2)
                distance[k] = addend1 + addend2 + addend3      
            labels[i] = np.argmin(distance)
    return labels