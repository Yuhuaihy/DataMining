from dataload import load_data
import numpy as np 
from sklearn import preprocessing
from sklearn import svm
from IPython import embed
# from svm import SVM
X_train, y_train, X_test, y_test = load_data()
label_y = np.where(y_train == 1)[1].reshape(len(y_train),1)
label_test = np.where(y_test == 1)[1].reshape(len(y_test), 1)
# label_y = [x for x in label_y]
# embed()
clf = svm.SVC(kernel = 'linear')
clf.fit(X_train, label_y)
score = clf.score(X_test, label_test)
print('The predict accuracy for Linear SVM is {}'.format(score))
#clf.predict(X_test)


