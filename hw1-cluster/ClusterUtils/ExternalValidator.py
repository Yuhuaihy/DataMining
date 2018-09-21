import pandas as pd

def find_norm_MI(true_labels, pred_labels):

    # Implement.
    # Return a number corresponding to the NMI of the two sets of labels.
    n = len(pred_labels)
    true_labels_counter = cl.Counter(true_labels)
    class_num = len(true_labels_counter)
    entropy_class = -1 * sum([(c/n) * math.log2(c/n) for c in true_labels_counter.values()])
    pred_labels_counter = cl.Counter(pred_labels)
    cluster_num = len(pred_labels_counter)
    entropy_cluster = -1 * sum([(c/n) * math.log2(c/n) for c in pred_labels_counter.values()])
    clusters = dict(zip(pred_labels_counter.keys(),range(cluster_num)))
    classes = dict(zip(true_labels_counter.keys(),range(class_num)))
    matrix = np.zeros((cluster_num,class_num))
    for i in range(n):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        k = clusters[pred_label]
        c = classes[true_label]
        matrix[k][c] += 1
    cluster_sum = matrix.sum(axis=1, dtype=float).reshape((cluster_num,1))
    p_matrix = matrix/cluster_sum   # probability pij
    c_matrix = cluster_sum/n      
    entropy_matrix = -1 * p_matrix * np.log2(p_matrix)
    entropy_matrix[np.isnan(entropy_matrix)] = 0
    entropy_cluster_class = entropy_matrix.sum(axis=1).reshape((cluster_num,1))
    entropy = (c_matrix * entropy_cluster_class).sum()
    mi = entropy_class-entropy
    nmi = 2*mi/(entropy_class+entropy_cluster)
    return nmi

def find_norm_rand(true_labels, pred_labels):

    # Implement.
    # Return a number corresponding to the NRI of the two sets of labels.
    n = len(pred_labels)
    cap_M = n*(n-1)/2
    true_labels_counter = cl.Counter(true_labels)
    class_num = len(true_labels_counter)
    pred_labels_counter = cl.Counter(pred_labels)
    cluster_num = len(pred_labels_counter)
    clusters = dict(zip(pred_labels_counter.keys(),range(cluster_num)))
    classes = dict(zip(true_labels_counter.keys(),range(class_num)))
    matrix = np.zeros((cluster_num,class_num))
    for i in range(n):
        pred_label = pred_labels[i]
        true_label = true_labels[i]
        k = clusters[pred_label]
        c = classes[true_label]
        matrix[k][c] += 1
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
    a = np.array(true_labels)
    b = np.array(pred_labels)
    sub = a-b
    zeros = sub[sub==0].size

    return zeros/total


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
