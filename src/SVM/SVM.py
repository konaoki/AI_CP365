import numpy as np
import pandas as pd

class SVM:
    def __init__(self):
        return None
    def fit(self,data,labels):
        nfeat=np.shape(data)[1]
        nex=np.shape(data)[0]
        lm=np.zeros(nex)
        lr = 0.01
        bias=0
        epochs=50
        for l in range(epochs):
            activation=0
            sum=0
            for i in range(nex):
                prediction=0
                for j in range(nex):
                    sum+=lm[j]*labels[j]*np.dot(data[i],data[j])
                of=1-0.5*labels[i]*sum
                #print(of)
                lm[i]=lm[i]+(lr/np.dot(data[i],data[i]))*of
            if l==epochs-1:
                biases=np.zeros(nex)
                for i in range(nex):
                    pl=self.predict(data[i],1,bias,lm,data,labels)
                    biases[i]=pl[0]
                bias=-1*np.average(biases)
        #print(pd.DataFrame(lm))
        self.bias=bias
        self.lm=lm
        self.data=data
        self.labels=labels
        weights=np.zeros(nfeat)
        for i in range(nex):
            weights=np.add(weights,lm[i]*labels[i]*data[i])
        self.weights=weights
    def predict(self, sample, raw=0, bias=None, lm=[], data=[], labels=[]):
        prediction=0
        if bias==None:
            bias=self.bias
        if len(lm)==0:
            lm=self.lm
        if len(data)==0:
            data=self.data
        if len(labels)==0:
            labels=self.labels
        for i in range(np.shape(data)[0]):
            prediction+=lm[i]*labels[i]*np.dot(data[i],sample)
        prediction+=bias
        if raw == 1:
            return [prediction,np.sign(prediction)]
        else:
            return np.sign(prediction)
