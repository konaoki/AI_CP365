import numpy as np
import pandas as pd
class OVA:
    def __init__(self, model):
        self.model=model
    def fit(self, data, labels):
        ulabels=np.unique(labels)
        for i,ul in enumerate(ulabels):
            
            self.model.fit(data,labels)
