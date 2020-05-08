import numpy as np
import numpy.random as nr
import pandas as pd
class DataGenerator:
    def __init__(self, d, n):
        self.d=d
        self.n=n
    def generate(self):
        n=self.n
        d=self.d
        dir = nr.randn(d)*100
        #print(dir)
        data = np.empty([n,d+1])
        for i in range(n):
            data[i,:d]=nr.random_integers(10,n/10)*nr.randn(d)
            if np.dot(dir,data[i,:d]) > 0:
                data[i,d]=1
            else:
                data[i,d]=-1
        return pd.DataFrame(data)
