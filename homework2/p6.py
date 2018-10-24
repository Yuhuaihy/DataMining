from dataload import load_data
import numpy as np 
from sklearn import preprocessing
from svm import SVM
X_train, y_train, X_test, y_test = load_data()
svm = SVM(X_train, y_train, X_test, y_test)
svm.fit()
svm.predict()