import numpy as np 
from IPython import embed

class SVM(object):
    def __init__(self, Xtrain, ytrain, Xtest, ytest):
        self.ytrain = ytrain
        self.m, self.classes = ytrain.shape
        bias = np.ones((1,self.m))
        self.Xtrain = np.insert(Xtrain, 0, values=bias, axis=1)
        _, self.n = self.Xtrain.shape
        self.Xtest = np.insert(Xtest, 0, values=bias, axis=1)
        self.ytest = ytest
        
        

    def fit(self):
        w = np.zeros((self.classes, self.n))
        eta = 1
        epochs = 100
        for epoch in range(1,epochs):
            for i in range(self.m):
                for j in range(self.classes):
                    y = self.ytrain[i][j]
                    if y * np.dot(self.Xtrain[i],w[j]) < 1:
                        w[j] += eta * (self.Xtrain[i] * y) + (-2 * (1/epochs) * w[j])
                    else:
                        w[j] = w[j] + eta * (-2  *(1/epoch)* w[j])
        self.w = w
        
    
    def predict(self, Xtest, ytest):
        predy = np.dot(X, self.w.transpose())
        embed()
    
    def plot(self):
        pass
    
    

    