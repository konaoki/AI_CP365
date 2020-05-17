import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from sklearn import svm

class MovieClassifier:
    print("Loading data...")
    data = pd.read_csv('../../movie_data.csv', header = None, skiprows=1,encoding='utf-8')

    nrow=1000
    print("Building bag of words...")
    #vectorize the entire corpus
    corpus = data.iloc[:nrow, 0].values
    vectorizer=CountVectorizer()
    vData = vectorizer.fit_transform(corpus).toarray() #bag of words
    print(np.shape(vData))


    step=int(nrow/10)
    error_rates=[]

    print("Testing...")
    for i in range(10):
        print("test #"+str(i))
        #Learning data
        pData = np.concatenate((vData[:i*step],vData[(i+1)*step:nrow]))
        labels = np.concatenate((data.iloc[:i*step, 1].values, data.iloc[(i+1)*step:nrow, 1].values))
        #test data
        testData = vData[i*step:(i+1)*step]
        testLabels = data.iloc[i*step:(i+1)*step, 1].values

        model = svm.SVC()
        model.fit(pData, labels)
        prediction = model.predict(testData)
        error=0
        for j in range(step):
            if prediction[j] != testLabels[j]:
                error+=1
        error_rates.append(error/step)
    #print(np.shape(error_rate))
    print(str(np.average(error_rates))+" +/- "+str(np.std(error_rates)/(len(error_rates)**0.5)))
