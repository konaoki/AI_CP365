import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import NeighborModel as nm
from sklearn.neighbors import KNeighborsClassifier

#url = os.path.join('https://archive.ics.uci.edu', 'ml','machine-learning-databases','iris','iris.data')
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, encoding='utf-8')
def normalize(): #min-max normalization
    nmd = data.copy()
    for feat in data.iloc[:150, 0:4].columns:
        #normalized_df=(df-df.mean())/df.std()
        max = data[feat].max()
        min = data[feat].min()
        nmd[feat] = (data[feat]-data[feat].mean())/data[feat].std()
    return nmd
#data = normalize()
#print(data)
pData = data.iloc[:150, 0:4].values
labels = data.iloc[:150,4].values
#binary label vector
labels = np.where( labels == 'Iris-virginica',1,-1)

#Make two data sets, one for training and one for testing (hold out test set)
#Standard method is 9:1 i.e training is 90% of the data and test is 10%
#Statistical procedure is 10-fold cross validation
ave_errorrates=[]
sk_ave_errorrates=[]

ks=[]
sk_ks=[]
for k in range(1,30):
    ks.append(k)
    sk_ks.append(k)

    totalerrorrate=0
    sk_totalerrorrate=0
    for i in range(0,10):
        stepsize=15
        testset = pData[stepsize*i:stepsize*(i+1),0:4]
        trainingset =  np.concatenate((pData[0:stepsize*i,0:4],pData[stepsize*(i+1):pData.shape[0],0:4]), axis=0)
        trainingsetlabels = np.concatenate((labels[0:stepsize*i],labels[stepsize*(i+1):pData.shape[0]]), axis=0)
        #print(trainingsetlabels)
        model = nm.NeighborModel(k)
        sk_model = KNeighborsClassifier(n_neighbors=k)

        model.fit(trainingset,trainingsetlabels)
        sk_model.fit(trainingset,trainingsetlabels)

        nerror=0
        sk_nerror=0
        for j in range(0,stepsize):
            gi= i*stepsize+j
            prediction = model.predict(testset[j])
            sk_prediction = sk_model.predict([testset[j]])
            if prediction!= labels[gi]:
                nerror+=1
            if sk_prediction!= labels[gi]:
                sk_nerror+=1
        errorrate = nerror/stepsize
        sk_errorrate = sk_nerror/stepsize

        totalerrorrate+=errorrate
        sk_totalerrorrate+=sk_errorrate

    ave_errorrates.append(totalerrorrate/10)
    sk_ave_errorrates.append(sk_totalerrorrate/10)


plt.scatter(ks,ave_errorrates)
#plt.scatter(sk_ks,sk_ave_errorrates)
plt.title("My KNN")
plt.xlabel("K value")
plt.ylabel("Average Error rate over 10 tests")
plt.show()
