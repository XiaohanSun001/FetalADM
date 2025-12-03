
import numpy as np

import pandas as pd

from sklearn import preprocessing
def dataStandard(unnor_data):
    standard = preprocessing.StandardScaler().fit(unnor_data)
    feature = [name for name in unnor_data.columns]
    nor_data = pd.DataFrame(standard.transform(unnor_data))
    nor_data.columns = feature
    return nor_data,standard
def traindataPrepare(X,y):
    nor_X,normalization = dataStandard(X)
    prepared_data = dataMerge(nor_X,y)
    np.random.seed(12)
    prepared_data = prepared_data.reindex(np.random.permutation(prepared_data.index))
    mean = normalization.mean_
    std = np.sqrt(normalization.var_)
    return prepared_data,mean,std

def testdataPrepare(x,y,mean,std):
    nor_X = (x-mean)/std
    prepared_data = dataMerge(nor_X,y)
    return prepared_data

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data



train_data = pd.read_csv(r'E:\sxh\DNA论文\lw6\mode_train_356\model\data_original0\Traindata_216_27.csv')
test_data = pd.read_csv(r'E:\sxh\DNA论文\lw6\mode_train_356\model\data_original0\Test_54_27.csv')
data,label = dataSplit(train_data)

trainset,mean,std = traindataPrepare(data,label)
print('....')
print(mean,std)

np.savetxt('traindata_mean.csv', mean, delimiter=',', fmt='%.9f')
np.savetxt('traindata_std.csv', std, delimiter=',', fmt='%.9f')

data1,label1 = dataSplit(test_data)
testset = testdataPrepare(data1, label1, mean, std)

#resultPath = 'Standardtraindata.xlsx'
resultPath_csv= 'Standardtraindata.csv'
#trainset.to_excel(resultPath,index = False,na_rep = 0,inf_rep = 0)
trainset.to_csv(resultPath_csv, index=False, na_rep=0, float_format='%.9f')

#resultPath1 = 'Standardtest.xlsx'
resultPath1_csv = 'Standardtest.csv'

#testset.to_excel(resultPath1,index = False,na_rep = 0,inf_rep = 0)
testset.to_csv(resultPath1_csv,index = False,na_rep = 0)