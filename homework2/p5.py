#Implement a KNN classifier in Python from scratch. 
# Test it using cross-validation (train:test = 4:1) on IRIS dataset 
# which you can obtain it from scikit-learn load iris function with K = 1, 2, 3. 
# Your program will be runned by ”python p5.py”
from IPython import embed
import numpy as np
from sklearn import datasets
from knn import Knn

iris = datasets.load_iris()
X = iris.data
y = iris.target
knn = Knn(X,y)
