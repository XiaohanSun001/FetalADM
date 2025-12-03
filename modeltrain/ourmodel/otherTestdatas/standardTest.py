
import numpy as np

import pandas as pd

from sklearn import preprocessing
def dataStandard(unnor_data):
    standard = preprocessing.StandardScaler().fit(unnor_data)
    feature = [name for name in unnor_data.columns]
    nor_data = pd.DataFrame(standard.transform(unnor_data))
    nor_data.columns = feature
    return nor_data,standard


def testdataPrepare(X,y,mean,std):
    nor_X = (X-mean)/std
    prepared_data = dataMerge(nor_X,y)
    print(X)
    print(mean)
    print(X-mean)
    print(std)
    return prepared_data

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data


def StandardTest(file):
    test_data = pd.read_csv(file+'.csv')
    data, label = dataSplit(test_data)
    testset = testdataPrepare(data, label, mean, std)

    resultPath = file+'_standard.csv'
    testset.to_csv(resultPath,index = False,na_rep = 0,float_format='%.9f')


def data_extract(filename, outname):
    #    feature_selet = ['label','MaxMin', 'SDchr13','SDchr18' ,'SDchr21','MW',
    #              'GW',	'AR', 'DR',
    #             'R_UR',	'GC' ,'SDchrX',	'SDchrY',  'ChrX', 'ChrY']

    feature_selet = ['label', 'Chr21', 'MaxMin', 'Zchr21', 'DR', 'Zchr18', 'Chr18', 'Zchr13', 'Chr13',
                     'ChrY', 'FY', 'R', 'R_UR', 'PT', 'GC']

    all_data = pd.read_csv(filename)
    data = all_data[feature_selet]

    data.to_csv(outname, index=False)


mean= np.loadtxt('traindata_mean_25.csv',delimiter=',')
std= np.loadtxt('traindata_std_25.csv',delimiter=',')

#print(mean)
#print(std)

filename1='1_25'
#filename2='T211_25'
#
StandardTest(filename1)
#StandardTest(filename2)
#
#
# data_extract(filename1+'_standard.csv', 'T136_14_standard.csv')
# data_extract(filename2+'_standard.csv', 'T211_14_standard.csv')