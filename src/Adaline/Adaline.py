import numpy as np
import pandas as pd

class Adaline:
    def __init__(self):
        return None
    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.nexamples = data.shape[0]
        self.nfeatures = data.shape[1]
        ws = np.ones((self.nfeatures,1))
        lr = 0.000001
        count=0
        tws= np.zeros((self.nfeatures,1))
        while np.sum(ws-tws)!=0: #if weights haven't reached equilibrium
        #while count<100:
            count+=1
            activation = np.dot(np.transpose(ws),np.transpose(data)) #row vector of activations by example
            diff = np.reshape(self.labels,(1,self.nexamples)) - activation #row vector of diff by examples
            dw = np.transpose(lr*np.dot(diff,data))
            tws=ws
            ws = ws + dw
        self.ws=ws
    def predict(self,sample):
        if np.dot(np.transpose(self.ws),sample) > 0:
            return 1
        else:
            return -1
    def copy(self):
        nmodel = Adaline()
        nmodel.ws=self.ws.copy()
        nmodel.data=self.data.copy()
        nmodel.labels=self.labels.copy()
        nmodel.nexamples=self.nexamples
        nmodel.nfeatures=self.nfeatures
        return nmodel
