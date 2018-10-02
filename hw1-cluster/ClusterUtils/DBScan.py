import pandas as pd
import numpy as np
import time
from ClusterUtils.SuperCluster import SuperCluster
from ClusterUtils.ClusterPlotter import _plot_generic_
from IPython import embed
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
    for i in range(m):
        distance_matrix[i] = ((X-X[i])**2).sum(axis=1)
    labels = np.zeros((m,1))
    c = 1
    for i in range(m):
        if labels[i] != 0:
            continue
        r = distance_matrix[i]
        neighbors = list(np.where(r<eps)[0])[1:]
        num = len(neighbors)
        if num<min_points:
            labels[i] = -1
            continue
        c += 1
        labels[i] = c
        for p in neighbors:
            expandCluster(c,neighbors,labels,eps,min_points,distance_matrix)
    return labels


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class DBScan(SuperCluster):
    """
    Perform DBSCAN clustering from vector array.
    Parameters
    ----------
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    min_samples : int, optional
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
    csv_path : str, default: None
        Path to file for dataset csv
    keep_dataframe : bool, default: True
        Hold on the results pandas DataFrame generated after each run.
        Also determines whether to use pandas DataFrame as primary internal data state
    keep_X : bool, default: True
        Hold on the results generated after each run in a more generic array-type format
        Use these values if keep_dataframe is False
    verbose: bool, default: False
        Optional log level
    """

    def __init__(self, eps=1, min_points=10, csv_path=None, keep_dataframe=True,
                                                    keep_X=True,verbose=False):
        self.eps = eps
        self.min_points = min_points
        self.verbose = verbose
        self.csv_path = csv_path
        self.keep_dataframe = keep_dataframe
        self.keep_X = keep_X

    # X is an array of shape (n_samples, n_features)
    def fit(self, X):
        if self.keep_X:
            self.X = X
        start_time = time.time()
        self.labels = dbscan(X, eps=self.eps, min_points = self.min_points,verbose = self.verbose)
        print("DBSCAN finished in  %s seconds" % (time.time() - start_time))
        return self

    def show_plot(self):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels)
        else:
            print('No data to plot.')

    def save_plot(self, name):
        if self.keep_dataframe and hasattr(self, 'DF'):
            _plot_generic_(df=self.DF, save=True, n=name)
        elif self.keep_X:
            _plot_generic_(X=self.X, labels=self.labels, save=True, n=name)
        else:
            print('No data to plot.')
