import numpy as np
import pandas as pd
class OVA:
    def __init__(self, model):
        self.model=model
    def fit(self, data, labels):
        self.data=data
        self.labels=labels
        label_lookup=dict()
        ulabels=np.unique(labels)
        self.ulabels=ulabels
        nfeat=len(ulabels)
        self.nfeat=nfeat
        models=[]
        for i,ul in enumerate(ulabels):
            label_lookup[ul]=i
            blabels = np.where(labels==ul,1,0) #binary labels
            self.model.fit(data,blabels)
            models.append(self.model.copy())
        self.label_lookup=label_lookup
        self.models=models
    def predict(self,sample):
        for i in range(self.nfeat):
            model=self.models[i]
            p=model.predict(sample)
            print(str(i)+" : "+str(p))
            #if p==1:
        #return self.ulabels[i]
        return 0
