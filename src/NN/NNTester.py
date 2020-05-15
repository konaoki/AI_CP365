import numpy as np
import pandas as pd
import Neuralnet as nn
class NNTester:
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header = None, encoding='utf-8')
    #data= data.replace(to_replace="Iris-setosa",value=0)
    #data=data.replace(to_replace="Iris-versicolor",value=1)
    #data=data.replace(to_replace="Iris-virginica",value=2)
    pData = data.iloc[:150, 0:4].values
    labels = data.iloc[:150,4].values
    labels = np.where( labels == 'Iris-virginica',1,-1)
    model=nn.Neuralnet(4,1,100)
    model.fit(pData,labels)

    print(model.predict([6.3,3.3,4.7,1.6]))
