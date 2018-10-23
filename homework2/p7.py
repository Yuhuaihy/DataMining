import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from dataload import load_data
from sklearn import preprocessing
from IPython import embed
h = .02  # step size in the mesh

X_train, y_train, X_test, y_test = load_data()
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)
# X_test_minmax = min_max_scaler.fit_transform(X_test)

scaler = preprocessing.StandardScaler().fit(X_train)
trainX = scaler.transform(X_train)
testX = scaler.transform(X_test)

names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "Neural Net","Naive Bayes"]

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    GaussianNB()]

for name, clf in zip(names, classifiers):
    clf.fit(trainX, y_train)
    score = clf.score(testX, y_test)
    embed()