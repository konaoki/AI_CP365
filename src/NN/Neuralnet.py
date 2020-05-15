import numpy as np
import pandas as pd
from numpy import random
class Neuralnet:
    def __init__(self, nhidden,nclass,nepoch):
        self.nhidden=nhidden
        self.nepoch=nepoch
        self.nclass=nclass
    def fit(self, data, labels):
        ndata=np.shape(data)[0]
        nfeature=np.shape(data)[1]
        nhidden=self.nhidden
        nclass=self.nclass
        v=random.rand(nfeature,nhidden)
        w=random.rand(nclass,nhidden)
        hlayer=np.zeros(nhidden)
        olayer=np.zeros(nclass)
        lr=0.01
        nepoch=self.nepoch
        print(v)
        print(w)
        print("------------")
        for epoch in range(nepoch):
            for d in range(ndata):
                dat=data[d]
                lab=labels[d]
                tempw=np.copy(w)
                for i in range(nfeature):
                    for j in range(nhidden):
                        dv=0
                        #calculate hidden layer nodes
                        hlayer[j]=0
                        for i2 in range(nfeature):
                            hlayer[j]+=dat[i2]*v[i2,j]
                        hlayer[j]=self.sigmoid(hlayer[j])
                        #constant
                        x=hlayer[j]*(1-hlayer[j])*dat[i]
                        #loop through outputs (usually one)
                        for r in range(nclass):
                            olayer[r]=0
                            for j2 in range(nhidden):
                                olayer[r]+=hlayer[j2]*w[r,j2]
                            olayer[r]=self.sigmoid(olayer[r])
                            dw=2*(olayer[r]-lab)*olayer[r]*(1-olayer[r])*hlayer[j]
                            #print(d)
                            #print(olayer[r])
                            #print(hlayer[j])
                            w[r,j]+=lr*dw
                            dv+=(olayer[r]-lab)*olayer[r]*(1-olayer[r])*tempw[r,j]*x
                        dv=dv*2
                        v[i,j]+=lr*dv
        print(v)
        print(w)
        self.v=v
        self.w=w
        self.hlayer=hlayer
        self.olayer=olayer
        self.data=data
        self.labels=labels
        self.nfeature=nfeature
        self.nhidden=nhidden
    def predict(self,sample):
        nfeature=self.nfeature
        nhidden=self.nhidden
        nclass=self.nclass
        output=np.zeros(nclass)
        for i in range(nclass):
            so=0
            for j in range(nhidden):
                sh=0
                for k in range(nfeature):
                    sh+=sample[k]*self.v[k,j]
                h=self.sigmoid(sh)
                so+=h*self.w[i,j]
            print(so)
            output[i]=self.sigmoid(so)
        return output
    def sigmoid(self, x):
        return 1/(1 + np.exp(-1*x))
