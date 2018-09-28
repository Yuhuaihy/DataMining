import pandas as pd
import numpy as np
import random
import math
import sys
from scipy.cluster.vq import kmeans2
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_
from IPython import embed
def build_simi_matrix(X):
    m = len(X)
    sigma = 2
    matrix = np.zeros((m,m))
    for i in range(m):
        matrix[i] = ((X-X[i]) ** 2).sum(axis=1).reshape((1,m))
    W = np.exp((-1/(2*(sigma**2)))* matrix)

    return W

def spectral(X, n_clusters=3, verbose=False):
    m = len(X)
    labels = np.zeros((m,1))
    simi_matrix = build_simi_matrix(X)
    d_matrix = np.sum(simi_matrix,axis=1)
    l = d_matrix - simi_matrix
    d2 = np.sqrt(1/d_matrix)
    d2 = np.diag(d2)
    
    lap_matrix = np.eye(m) - np.dot((np.dot(d2,l)),d2)
    U,s,V = np.linalg.svd(lap_matrix,full_matrices=True)
    kerN = U[:,m-n_clusters+1:m]	
    for i in range(m):		
        kerN[i,:] = kerN[i,:] / np.linalg.norm(kerN[i,:])	
    _,labels = kmeans2(kerN,n_clusters,iter=100)
    return labels
##numpy.linalg.eig


    # Implement.

    # Input: np.darray of samples

    # Return an array or list-type object corresponding to the predicted
    # cluster numbers, e.g., [0, 0, 0, 1, 1, 1, 2, 2, 2]



# Add parameters below as needed, depending on your implementation.
# Explain your reasoning in the comments.

class Spectral(SuperCluster):

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
        self.labels = spectral(X, n_clusters=self.n_clusters, verbose=self.verbose)
        print("Spectral clustering finished in  %s seconds" % (time.time() - start_time))
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
