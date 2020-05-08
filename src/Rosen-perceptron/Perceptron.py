import numpy as np
import pandas as pd

class Perceptron:
    def __init__(self):
        return None
    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.nexamples = data.shape[0]
        self.nfeatures = data.shape[1]

        ws = np.empty(self.nfeatures)
        lr = 1
        errorrate = 1
        count = 0
        while errorrate != 0:
            count+=1
            error=0
            for i in range(self.nexamples):
                diff = self.labels[i] - self.predict(data[i],ws)
                ws = ws + lr*(diff)*data[i]
                if diff!=0:
                    error+=1
            errorrate = error/self.nexamples
            if count>1000/lr:
                count=0
                lr=lr/10
                #print(lr)

        self.ws=ws
    def predict(self, sample, ws):
        prediction = np.dot(np.transpose(ws),sample)
        prediction = np.where( prediction >=0,1,-1)
        return prediction
    def getWeights(self):
        return self.ws
