import pandas as pd
import numpy as np
class NeighborModel:
    def __init__(self,k):
        self.k=k
        return None
    def fit(self, data,labels):
        self.data=data
        self.labels=labels
        self.nexamples=data.shape[0]
        self.nfeatures=data.shape[1]

    def getDistance(self,d1,d2): #returns euclidean distance between two datum
        dist=0.0
        for i in range(self.nfeatures):
            dist+=(d2[i]-d1[i])**2
        return np.sqrt(dist)
    def getNeighbors(self,d):
        dist_labels = [] #distance, label
        for i in range(0,self.nexamples):
            item = []
            item.append(self.getDistance(d,self.data[i]))
            item.append(self.labels[i])
            dist_labels.append(item)

        dist_labels_df = pd.DataFrame(dist_labels)
        dist_labels_df=dist_labels_df.sort_values(by=0)
        #print("distances: "+str(dist_labels_df.iloc[0:self.k,1].values))
        return dist_labels_df.iloc[0:self.k,1].values
    def predict(self,d):
        nbs = self.getNeighbors(d)
        #print(nbs)
        sum=np.sum(nbs)
        #print(np.sum(nbs,axis=0))
        if sum> 0:
            return 1
        else:
            return -1
