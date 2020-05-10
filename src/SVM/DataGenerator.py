import numpy as np
import numpy.random as nr
import pandas as pd
class DataGenerator:
    def __init__(self, d, n,percent_error):
        self.d=d
        self.n=n
        self.percent_error=percent_error
    def generate(self):
        n=self.n
        d=self.d
        dir = nr.randn(d)
        separation=10
        data = np.empty([n,d+1])
        nerror=round(n*self.percent_error)
        for i in range(n):
            data[i,:d]=nr.random_integers(1,n/10)*nr.randn(d)
            sign=np.sign(np.dot(dir,data[i,:d]))
            data[i,:d]=np.add(data[i,:d],sign*dir*separation)
            data[i,d]=sign
        for i in range(n-nerror,n):
            data[i,:d]=nr.random_integers(1,n/20)*nr.randn(d)
            sign=0
            if nr.random()>0.5:
                sign=1
            else:
                sign=-1
            data[i,d]=sign

        return pd.DataFrame(data)
