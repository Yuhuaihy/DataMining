import pandas as pd
import time
from ClusterUtils.ClusterPlotter import _plot_cvnn_
from ClusterUtils.ClusterPlotter import _plot_silhouette_
import numpy as np
def tabulate_silhouette(datasets, cluster_nums):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.(By default, the datasets generated are pandas DataFrames, 
    # and the final column is named 'CLUSTER')
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]
    silhouette = []
    for idx in range(len(datasets)):
        #data = dataset.values[:,:-1]
        #label = dataset.values[:,-1]
        dataset = datasets[i]
        data = dataset[:,:-1]
        label = dataset[:,-1]
        n = len(data)
        matrix = np.zeros((n,n))
        for i in range(n):
            matrix[i] = ((data - data[i]) * (data-data[i])).sum(axis=1)
        clusters = {}
        for i in range(n):
            l = label[i]
            if l not in clusters:
                clusters[l] = []
            clusters[l].append(i)
        s_matrix = np.zeros((n,1))
        for i in range(n):
            l = label[i]
            a = 0
            b = []
            for k in clusters:
                points = clusters[k]
                r = matrix[i][points]
                size = len(r)
                if k==l:
                    a = r.sum()/(size-1)
                else:
                    b.append(r.sum()/size)
            b = min(b)
            s = (b-a)/max(a,b)
            s_matrix[i] = s
        s_total = 0
        for k in clusters:
            points = clusters[k]
            s = s_matrix[points]
            s_mean_cluster = np.mean(s)
            s_total += s_mean_cluster
        s = s_total/cluster_nums[idx]
        silhouette.append(s)
    dfdata = {'CLUSTERS':cluster_nums,'SILHOUETTE_IDX':silhouette}
    df = pd.DataFrame(dfdata)


    # Return a pandas DataFrame corresponding to the results.

    return df

def tabulate_cvnn(datasets, cluster_nums, k_vals):

    # Implement.

    # Inputs:
    # datasets: Your provided list of clustering results.
    # cluster_nums: A list of integers corresponding to the number of clusters
    # in each of the datasets, e.g.,
    # datasets = [np.darray, np.darray, np.darray]
    # cluster_nums = [2, 3, 4]
    
    # Return a pandas DataFrame corresponding to the results.
    for i in range(len(cluster_nums)):
        dataset = datasets[i]
        

    return None


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class InternalValidator:
    """
    Parameters
    ----------
    datasets : list or array-type object, mandatory
        A list of datasets. The final column should cointain predicted labeled results
        (By default, the datasets generated are pandas DataFrames, and the final
        column is named 'CLUSTER')
    cluster_nums : list or array-type object, mandatory
        A list of integers corresponding to the number of clusters used (or found).
        Should be the same length as datasets.
    k_vals: list or array-type object, optional
        A list of integers corresponding to the desired values of k for CVNN
        """

    def __init__(self, datasets, cluster_nums, k_vals=[1, 5, 10, 20]):
        self.datasets = datasets
        self.cluster_nums = cluster_nums
        self.k_vals = k_vals

    def make_cvnn_table(self):
        start_time = time.time()
        self.cvnn_table = tabulate_cvnn(self.datasets, self.cluster_nums, self.k_vals)
        print("CVNN finished in  %s seconds" % (time.time() - start_time))

    def show_cvnn_plot(self):
        _plot_cvnn_(self.cvnn_table)

    def save_cvnn_plot(self, name='cvnn_plot'):
        _plot_cvnn_(self.cvnn_table, save=True, n=name)

    def make_silhouette_table(self):
        start_time = time.time()
        self.silhouette_table = tabulate_silhouette(self.datasets, self.cluster_nums)
        print("Silhouette Index finished in  %s seconds" % (time.time() - start_time))

    def show_silhouette_plot(self):
        _plot_silhouette_(self.silhouette_table)

    def save_silhouette_plot(self, name='silhouette_plot'):
        _plot_silhouette_(self.cvnn_table, save=True, n=name)

    def save_csv(self, cvnn=False, silhouette=False, name='internal_validator'):
        if cvnn is False and silhouette is False:
            print('Please pass either cvnn=True or silhouette=True or both')
        if cvnn is not False:
            filename = name + '_cvnn_' + (str(round(time.time()))) + '.csv'
            self.cvnn_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if silhouette is not False:
            filename = name + '_silhouette_' + (str(round(time.time()))) + '.csv'
            self.silhouette_table.to_csv(filename)
            print('Dataset saved as: ' + filename)
        if cvnn is False and silhouette is False:
            print('No data to save.')
