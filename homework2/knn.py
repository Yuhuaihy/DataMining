#Implement a KNN classifier in Python from scratch. 
# Test it using cross-validation (train:test = 4:1) on IRIS dataset 
# which you can obtain it from scikit-learn load iris function with K = 1, 2, 3. 
# Your program will be runned by ”python p5.py”
from IPython import embed
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import random
import collections



class Knn(object):
    def __init__(self, data, y, ks=[1,2,3]):
        self.data = data
        self.y = y
        self.ks = ks
    
    def __generateData(self):
        n = len(self.y)
        each = n // 5
        testindex = random.sample(list(range(n)), each)
        self.trainX = []
        self.trainy = []
        self.testX = []
        self.testy = []
        for i in range(n):
            if i in testindex:
                self.testX.append(self.data[i])
                self.testy.append(self.y[i])
            else:
                self.trainX.append(self.data[i])
                self.trainy.append(self.y[i])
        self.trainX = np.array(self.trainX)
        self.trainy = np.array(self.trainy)
        self.testX = np.array(self.testX)
        self.testy = np.array(self.testy).reshape((each, 1))
        
        
    

    def __distance(self, Xtrain, Xtest):
        n = len(Xtest)
        m = len(Xtrain)
        distance = np.zeros((n,m))
        for i in range(n):
            distance[i] = (((Xtrain - Xtest[i])**2).sum(axis=1)).reshape((1,m))
        return distance




    def __knn(self, k):
        n = len(self.testX)
        pred_y = np.zeros((n,1))
        distance = self.__distance(self.trainX, self.testX)
        dis_sort = np.argsort(distance)
        for i in range(n):
            min_idxs = dis_sort[i][:k]
            labels = [self.trainy[i] for i in min_idxs]
            if k == 1 or k == 2:
                label = labels[0]
            elif k == 3:
                label = labels[2] if labels[2] == labels[1] else labels[0]
            else:
                counter = collections.Counter(labels)
                max_count = 0
                for x in counter:
                    if counter[x] > max_count:
                        label = x
                        max_count = counter[x]
            pred_y[i] = label
        return pred_y
    
    def classify(self):
        self.__generateData()
        for k in self.ks:
            pred_y = self.__knn(k)
            sub = pred_y - self.testy
            correct = sub[sub==0].shape[0]
            accuracy = 1.0 * correct / len(pred_y)
            print('When k is %d, the prediction accuracy is %f'%(k, accuracy))

            





