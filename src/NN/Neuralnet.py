import numpy as np
import pandas as pd
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
        v=np.zeros(nfeature,nhidden)
        w=np.zeros(nclass,nhidden)
        hlayer=np.zeros(nhidden)
        olayer=np.zeros(nfeature)
        nepoch=self.nepoch
        for epoch in range(nepoch):
            for d in range(ndata):
                dat=data[d]
                lab=labels[d]
                for i in range(nfeature):
                    for j in range(nhidden):
                        dw=2*(olayer[i]-lab)*olayer[i]*(1-olayer[i])*hlayer[j]
                        dv=0
                        x=hlayer[j]*(1-hlayer[j])*dat[i]
                        for r in range(nclass):
                            dv+=(olayer[r]-lab)*olayer[r]*(1-olayer[r])*w[r,j]*x
                        dv=dv*2
                        w[i,j]+=dw
                        v[i,j]+=dv
        self.v=v
        self.w=w
        self.hlayer=hlayer
        self.olayer=olayer
        self.data=data
        self.labels=labels
    def predict(self,sample):
        nfeature=self.nfeature
        nhidden=self.nhidden
        output=np.zeros(nfeature)
        for i in range(nclass):
            so=0
            for j in range(nhidden):
                sh=0
                for k in range(nfeature):
                    sh+=sample[k]*self.v[k,j]
                h=1/(1 + np.exp(-sh))
                so+=h*self.w[i,j]
            output[i]=1/(1 + np.exp(-so))
        return output
