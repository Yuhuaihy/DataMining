import pandas as pd
import math
import collections as cl
import numpy as np
import time
from IPython import embed
import random
def build_matrix(true_labels, pred_labels):
    n = len(pred_labels)
    pred = set(pred_labels)
    cluster_num = len(pred)
    tru = set(true_labels)
    class_num = len(tru)
    clusters = dict(zip(pred,range(cluster_num)))
    classes = dict(zip(tru,range(class_num)))
    matrix = np.zeros((cluster_num,class_num))
    for i in range(n):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        k = clusters[pred_label]
        c = classes[true_label]
        matrix[k][c] += 1
    return matrix

def find_class_cluster_entropy(matrix, label):
    n = matrix.sum()
    if label == 'class':
        sum_matrix = matrix.sum(axis=0)
    else:
        sum_matrix = matrix.sum(axis=1)
    p_matrix = sum_matrix/n
    entropy_matrix = -1 * p_matrix * np.log2(p_matrix)
    return entropy_matrix.sum()

def find_entropy(matrix):
    cluster_num, class_num = matrix.shape
    cluster_sum = matrix.sum(axis=1, dtype=float).reshape((cluster_num,1))
    p_matrix = matrix/cluster_sum   # probability pij
    c_matrix = cluster_sum/matrix.sum()    
    entropy_matrix = -1 * p_matrix * np.log2(p_matrix)
    entropy_matrix[np.isnan(entropy_matrix)] = 0
    entropy_cluster_class = entropy_matrix.sum(axis=1).reshape((cluster_num,1))
    entropy = (c_matrix * entropy_cluster_class).sum()
    return entropy

def find_norm_MI(true_labels, pred_labels):
    matrix = build_matrix(true_labels, pred_labels) 
    entropy = find_entropy(matrix)
    entropy_class = find_class_cluster_entropy(matrix, 'class')
    entropy_cluster = find_class_cluster_entropy(matrix, 'cluster')
    mi = entropy_class-entropy
    nmi = 2*mi/(entropy_class+entropy_cluster)
    return nmi

def find_norm_rand(true_labels, pred_labels):

    # Implement.
    # Return a number corresponding to the NRI of the two sets of labels.
    n = len(pred_labels)
    cap_M = n*(n-1)/2
    matrix = build_matrix(true_labels, pred_labels)
    m = (matrix * (matrix-1) / 2.0).sum()
    matrix_cluster = matrix.sum(axis=1)
    m1 = (matrix_cluster * (matrix_cluster-1) /2.0).sum()
    matrix_class = matrix.sum(axis=0)
    m2 = (matrix_class * (matrix_class-1) /2.0).sum()
    nr = (m-(m1*m2)/cap_M)/(m1/2+m2/2-(m1*m2)/cap_M)
    return nr

def find_accuracy(true_labels, pred_labels):

    # Implement.
    # Return a number corresponding to the accuracy of the two sets of labels.
    total = len(true_labels)
    matrix = build_matrix(true_labels, pred_labels)
    n_cluster, n_class = matrix.shape
    correct = 0
    if n_cluster < n_class:
        matrix = np.transpose(matrix)
    for row in matrix:
        idx = np.argmax(row)
        


    return 1


# The code below is completed for you.
# You may modify it as long as changes are noted in the comments.

class ExternalValidator:
    """
    Parameters
    ----------
    df : pandas DataFrame, optional
        A DataFrame produced by running one of your algorithms.
        The relevant labels are automatically extracted.
    true_labels : list or array-type object, mandatory
        A list of strings or integers corresponding to the true labels of
        each sample
    pred_labels: list or array-type object, optional
        A list of integers corresponding to the predicted cluster index for each
        sample
    """

    def __init__(self, df = None, true_labels = None, pred_labels = None):
        df = df.drop('CENTROID', axis=0)  # IMPORTANT -- Drop centroid rows before processing
        self.DF = df
        self.true_labels = true_labels
        self.pred_labels = pred_labels

        if df is not None:
            self.extract_labels()
        elif true_labels is None or pred_labels is None:
            print('Warning: No data provided')

    def extract_labels(self):
        self.true_labels = self.DF.index
        self.pred_labels = self.DF['CLUSTER']

    def normalized_mutual_info(self):
        start_time = time.time()
        nmi = find_norm_MI(self.true_labels, self.pred_labels)
        print("NMI finished in  %s seconds" % (time.time() - start_time))
        return nmi

    def normalized_rand_index(self):
        start_time = time.time()
        nri = find_norm_rand(self.true_labels, self.pred_labels)
        print("NMI finished in  %s seconds" % (time.time() - start_time))
        return nri

    def accuracy(self):
        start_time = time.time()
        a = find_accuracy(self.true_labels, self.pred_labels)
        print("Accuracy finished in  %s seconds" % (time.time() - start_time))
        return a
