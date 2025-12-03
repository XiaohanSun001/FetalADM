
"""
Created on Fri Mar  5 00:34:25 2021

@author: Dell
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import sys, os
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import sys
from sklearn.metrics import make_scorer , accuracy_score

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y

def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data

def performance(y_true, y_pred):

    
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    fl_micro = f1_score(y_true, y_pred, average='micro')
    fl_weighted = f1_score(y_true, y_pred, average='weighted')

    
    mcc = matthews_corrcoef(y_true, y_pred)
    
    m=[accuracy, \
           precision_micro, precision_weighted, \
           recall_micro, recall_weighted, \
           fl_micro, fl_weighted,mcc]
    metric=[round(i,3) for i in m] 

    return metric

def prediction(data):
    model1 = joblib.load("step1/model1.pkl")
    y_pred_m1 = model1.predict(data)
    index1 = []
    index2 = []
    a = 0
    for i in y_pred_m1:
        # if aneuploid
        if i != 0:
            index1.append(a)
            a = a + 1
        # if euploid
        else:
            index2.append(a)
            a = a + 1

    y_pred = ['F'] * len(y_pred_m1)
    # print(y_pred)
    for i in index2:
        y_pred[i] = 0

    if len(index1) != 0:
        selet1 = test_data.loc[index1]
        data1, label1 = dataSplit(selet1)
        model2 = joblib.load("step2/model2.pkl")
        y_pred_m2 = model2.predict(data1)
        y_pred_m2 = y_pred_m2 + 1
        n = 0
        for i in index1:
            y_pred[i] = y_pred_m2[n]
            n = n + 1
    pred_label = y_pred
    return pred_label

if __name__ == "__main__":


    file_name = 'T210_14_standard.csv'

    #file_name ='test54_14.csv'
    outdir = "./Results/"
    outname=file_name[:-4]

    test_data = pd.read_csv(file_name)
    #print(test_data.head())
    
    prid_data, prid_label = dataSplit(test_data)
    
    pred_label = prediction(prid_data)



    per_test=performance(prid_label, pred_label)


    print('Performance on Test dataset '+file_name)
    print('accuracy,precision_micro, precision_weighted, recall_micro, recall_weighted, fl_micro, fl_weighted,mcc')
    print(per_test)

    cm = confusion_matrix(prid_label, pred_label)
    print(cm)

    out=open('Test_results.txt','a')

    out.write(file_name+'\n')
    out.write('Test Set Performance: \n')
    out.write('accuracy, precision_micro, precision_weighted, recall_micro, recall_weighted,  fl_micro, fl_weighted,mcc,specificity \n')
    #out.write(f'{accuracy}, {precision_micro}, {precision_weighted}, {recall_micro}, {recall_weighted},  {fl_micro}, {fl_weighted} \n')
    out.write(f'{per_test} \n')
    out.write(f'{cm} \n')

    out.write(f'\n')


    out.close()

    prid_label=np.array(prid_label).reshape(-1,1)
    pred_label=np.array(pred_label).reshape(-1,1)
    result = pd.DataFrame(np.hstack((prid_label,pred_label)))
    result.columns = ['True_label','Pred_label']
    result.to_csv(outdir+outname+'_result.csv',index=False)



