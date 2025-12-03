
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from sklearn import preprocessing


def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data
t21_data = pd.read_csv('T21.csv')
t18_data = pd.read_csv('T18.csv')
t13_data = pd.read_csv('T13.csv')

data,label = dataSplit(t21_data)
data1,label1 = dataSplit(t18_data)
data2,label2 = dataSplit(t13_data)

sm = SMOTE(random_state=27)
print('T21 dataset shape %s'% Counter(label))
X_t21, Y_t21 = sm.fit_resample(data,label )
print('T21 Resampled dataset shape %s' % Counter(Y_t21))
prepared_data = dataMerge(X_t21, Y_t21)
resultPath = 'smoteT21.xlsx'
prepared_data.to_excel(resultPath,index = False,na_rep = 0,inf_rep = 0)

print('T18 dataset shape %s'% Counter(label1))
X_t18, Y_t18 = sm.fit_resample(data1,label1 )
print('T18 Resampled dataset shape %s' % Counter(Y_t18))
prepared_data = dataMerge(X_t18, Y_t18)
resultPath = 'smoteT18.xlsx'
prepared_data.to_excel(resultPath,index = False,na_rep = 0,inf_rep = 0)

sm1 = SMOTE(random_state=27,k_neighbors=2)
print('T13 dataset shape %s'% Counter(label2))
X_t13, Y_t13 = sm1.fit_resample(data2,label2 )
print('T13 Resampled dataset shape %s' % Counter(Y_t13))
prepared_data = dataMerge(X_t13, Y_t13)
resultPath = 'smoteT13.xlsx'
prepared_data.to_excel(resultPath,index = False,na_rep = 0,inf_rep = 0)
