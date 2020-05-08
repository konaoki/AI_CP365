import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Perceptron
import DataGenerator
class PerceptronTest:
    stepsize=15
    ndata = 10*stepsize
    nfeat = 4
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, encoding='utf-8')
    #dg = DataGenerator.DataGenerator(nfeat,ndata)
    #data = dg.generate()
    #data.to_csv('10d_test_sparse.data', index=False,header=False)
    #data=pd.read_csv('test_tight.data', header = None, encoding='utf-8')
    #print(data)
    pData = data.iloc[:ndata, :nfeat].values
    labels = data.iloc[:ndata,nfeat].values

    labels = np.where( labels == 'Iris-setosa',1,-1)


    model = Perceptron.Perceptron()
    totalerrorrate=0
    for i in range(10):
        testset = pData[stepsize*i:stepsize*(i+1),0:nfeat]
        trainingset =  np.concatenate((pData[0:stepsize*i,0:nfeat],pData[stepsize*(i+1):pData.shape[0],0:nfeat]))
        trainingsetlabels = np.concatenate((labels[0:stepsize*i],labels[stepsize*(i+1):pData.shape[0]]))
        model.fit(trainingset,trainingsetlabels)
        nerror=0
        for j in range(0,stepsize):
            gi= i*stepsize+j
            prediction = model.predict(testset[j],model.getWeights())
            if prediction!= labels[gi]:
                nerror+=1
        errorrate = nerror/stepsize
        totalerrorrate+=errorrate
        print("test "+str(i)+": "+str(errorrate))
    print("average error rate over 10 tests"+str(totalerrorrate/10))
    #plt.scatter(pData[:,0],pData[:,1],c=labels)
    #plt.show()
