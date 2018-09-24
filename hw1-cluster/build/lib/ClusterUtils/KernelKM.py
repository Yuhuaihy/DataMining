import pandas as pd
import numpy as np
import random
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_
def GaussianKernel(X):
    n = len(X)
    sigma = 2
    kernel_matrix = np.zeros((n,n))
    for i in range(n):
        exp = (-1/(2*(sigma**2)) * (X-X[i])**2).sum(axis=1)
        kernel_matrix[i] = np.power(np.e,exp.reshape((n,1)))
    return kernel_matrix

def kernel_km(X,n_clusters=3, verbose=False):
    kernel_matrix = GaussianKernel(X)
    m = len(X)
    centroids = random.sample(range(n_clusters),n_clusters)
    labels = [0] * n_clusters
    for x in centroids:
        labels[x] = x
    max_iter = 300

    ### distance = k(xi,xi)^2 +2* sum(k(xi,xj))/c + sum(k(xj,xl))/c^2
    ### addend1+addend2+addend3
    for _ in range(max_iter):
        distance = np.zeros((n_clusters,1))
        for i in range(m):
            for k in range(n_clusters):
                points = np.where(labels==k)[0]
                num = len(points)
                addend1 = kernel_matrix[i][i] ** 2
                r1 = kernel_matrix[i][:]
                r2 =r1[points]
                addend2 = 2 * r2.sum() / num
                r3 = kernel_matrix[points]
                r3 = r3[:,points]
                addend3 = r3.sum() / (num**2)
                distance[k] = addend1 + addend2 + addend3
            labels[i] = np.argmin(distance)

                




    # Implement.

    # Input: np.darray of samples

    # Return an array or list-type object corresponding to the predicted
    # cluster numbers, e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]

    return labels


# Add parameters below as needed, depending on your implementation.
# Explain your reasoning in the comments.

class KernelKM(SuperCluster):

    def __init__(self, n_clusters=3, csv_path=None, keep_dataframe=True,
                                            keep_X=True, verbose=False):
        self.n_clusters=n_clusters
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X
        self.verbose = verbose

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels = kernel_km(X, n_clusters=self.n_clusters, verbose=self.verbose)
        print("KernelKM finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels)
        else:
            print('No data to plot.')

    def save_plot(self, name='spectral_plot'):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels, save=True, n=name)
        else:
            print('No data to plot.')