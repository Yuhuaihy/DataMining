###SVM
from svm import SVM
import numpy as np
from IPython import embed
import scipy.io as sio
def load_data():
    trainX_path = 'datasets/train_data.mat'
    trainy_path = 'datasets/train_label.mat'
    testX_path = 'datasets/test_data.mat'
    testy_path = 'datasets/test_label.mat'
    mat_content = sio.loadmat(trainX_path)
    trainX = mat_content['train_data'].transpose()
    m, n = trainX.shape
    mat_content = sio.loadmat(trainy_path)
    trainy = mat_content['train_label'].transpose()
    train_label = np.where(trainy == 1)[1].reshape((m,1))

    mat_content = sio.loadmat(testX_path)
    testX = mat_content['test_data'].transpose()
    m2, n2 = testX.shape
    mat_content = sio.loadmat(testy_path)
    testy = mat_content['test_label'].transpose()
    test_label = np.where(testy == 1)[1].reshape((m2,1))
    return trainX, trainy, testX, testy

 

