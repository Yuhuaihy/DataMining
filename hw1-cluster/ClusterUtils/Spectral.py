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
from numpy import linalg as LA
import scipy.sparse as sps
from sklearn.cluster import KMeans
def build_simi_matrix(X):
    m = len(X)
    sigma = 2
    matrix = np.zeros((m,m))
    for i in range(m):
        matrix[i] = ((X-X[i]) ** 2).sum(axis=1).reshape((1,m))
    W = np.exp((-1/(2*(sigma**2)))* matrix)
    W = W - np.diag(np.diag(W))
    return W

# def build_lap(A):
#     D = np.zeros(A.shape)
#     w = np.sum(A, axis=0)
#     D.flat[::len(w) + 1] = w ** (-0.5)  # set the diag of D to w
#     return D.dot(A).dot(D)
    
# def getEigVec(Lap_matrix, n_cluster):
#     val_prop, vect_prop = sps.linalg.eigs(Lap_matrix, n_cluster)
#     X = vect_prop.real    
#     rows_norm = np.linalg.norm(vect_prop.real, axis=1, ord=2)
#     return (X.T / rows_norm).T

    # val, vec = LA.eig(Lap_matrix)
    # dim = len(val)
    # dictval = dict(zip(val, range(dim)))
    # keig = np.sort(val)[:n_cluster]
    # idx = [dictval[k] for k in keig]
    # return vec[:,idx]

def spectral(X, n_clusters=3, verbose=False):
    m = len(X)
    labels = np.zeros((m,1))
    simi_matrix = build_simi_matrix(X)
    d_matrix = np.sum(simi_matrix,axis=1)
    d2 = np.sqrt(1/d_matrix)
    d2 = np.diag(d2)
    lap_matrix = np.dot((np.dot(d2,simi_matrix)),d2)

    U,s,V = np.linalg.svd(lap_matrix,full_matrices=True)
    kerN = U[:,m-n_clusters+1:]	
    for i in range(m):		
        kerN[i,:] = kerN[i,:] / np.linalg.norm(kerN[i,:])	
    _,labels = kmeans2(kerN,n_clusters,iter=100)
    
    return labels
##numpy.linalg.eig





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
