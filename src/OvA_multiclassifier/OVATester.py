import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../SVM')
sys.path.insert(1, '../Adaline')
import SVM
import Adaline
import OVA
from sklearn.datasets import load_digits
class OVATester:
    digits = load_digits()
    svm = SVM.SVM()
    adaline = Adaline.Adaline()
    ova = OVA.OVA(svm)
    nex=1000
    ova.fit(digits.data[:nex],digits.target[:nex])
    error=0
    for i in range(15):
        print(digits.target[nex+i])
        if ova.predict(digits.data[nex+i])!=digits.target[nex+i]:
            error+=1
    print(error)
