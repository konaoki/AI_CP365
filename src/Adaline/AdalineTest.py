import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Adaline
import DataGenerator
class AdalineTest:
    ndata = 150
    nfeat = 4
    #dg = DataGenerator.DataGenerator(nfeat,ndata)
    #data = dg.generate()
    #data.to_csv('4d_test.data', index=False,header=False)
    #data=pd.read_csv('4d_test.data', header = None, encoding='utf-8')
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, encoding='utf-8')
    data = data.sample(frac=1)
    pData = data.iloc[:ndata, :nfeat].values
    labels = data.iloc[:ndata,nfeat].values
    labels = np.where( labels == 'Iris-virginica',1,-1)

    model = Adaline.Adaline()
    errorrate=0
    for i in range(10):
        error=0
        trainingData=np.concatenate((pData[:15*i,:],pData[15*(i+1):,:]))
        training_labels=np.concatenate((labels[:15*i],labels[15*(i+1):]))
        testData=pData[15*i:15*(i+1),:]
        model.fit(trainingData,training_labels)
        for j in range(15):
            p = model.predict(testData[j,:])
            if p!=labels[i*15+j]:
                error+=1
        print("error rate: "+str(error/15))
        errorrate+=error/15
    print("average error rate: "+str(errorrate/10))
