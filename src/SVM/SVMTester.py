import numpy as np
import pandas as pd
import DataGenerator
import SVM
import matplotlib.pyplot as plt
class SVMTester:
    ndata = 300
    nfeat = 2
    dg = DataGenerator.DataGenerator(nfeat,ndata,0.1)
    data = dg.generate()
    data.to_csv('2d_test_err.data', index=False,header=False)
    #data=pd.read_csv('2d_test_err.data', header = None, encoding='utf-8')
    pData = data.iloc[:ndata, :nfeat].values
    labels = data.iloc[:ndata,nfeat].values

    model=SVM.SVM()
    model.fit(pData,labels)

    nerror=0
    for i in range(ndata):
        pl=model.predict(pData[i])
        if pl!=labels[i]:
            nerror+=1

    print(nerror)
    ws=model.weights
    print(ws)
    plt.scatter(pData[:,0],pData[:,1],c=labels)
    x = np.linspace(np.min(pData[:,0]), np.max(pData[:,0]), 1000)
    plt.plot(x,-x*ws[0]/ws[1])
    plt.quiver(ws[0]/np.linalg.norm(ws),ws[1]/np.linalg.norm(ws),color='g',scale=10)
    plt.axis([np.min(pData[:,0]), np.max(pData[:,0]), np.min(pData[:,1]), np.max(pData[:,1])])
    plt.show()
