#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 7 00:34:25 2021

@author: xiaohansun
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,auc
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import  OneHotEncoder
import matplotlib.pyplot as plt
import shap
a = 3

# normalization
def dataStandard(unnor_data):
    standard = preprocessing.StandardScaler().fit(unnor_data)
    feature = [name for name in unnor_data.columns]
    nor_data = pd.DataFrame(standard.transform(unnor_data))
    nor_data.columns = feature
    return nor_data,standard

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y

def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data

# data processing

# model performance
def performance(y_true, y_pred):

    accuracy=accuracy_score(y_true, y_pred)
    precision_micro=precision_score(y_true, y_pred, average='micro')
    precision_weighted=precision_score(y_true, y_pred, average='weighted')
    recall_micro=recall_score(y_true, y_pred, average='micro')
    recall_weighted=recall_score(y_true, y_pred, average='weighted')
    fl_micro = f1_score(y_true, y_pred, average='micro')
    fl_weighted = f1_score(y_true, y_pred, average='weighted')

    return accuracy, \
           precision_micro,precision_weighted,\
           recall_micro,recall_weighted,\
           fl_micro,fl_weighted

def model_evaluation(data,label):

    Accuracy=[]
    Precision_micro=[]
    Precision_weighted=[]
    Recall_micro=[]
    Recall_weighted=[]
    Fl_micro=[]
    Fl_weighted=[]
    
    # AUC = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(5,5))
    m=1

    kfold = StratifiedKFold(n_splits = a, shuffle=True, random_state =5)
    for train_index, validation_index in kfold.split(data, label):
        X_train, X_validation = data.iloc[train_index], data.iloc[validation_index]
        y_train, y_validation = label.iloc[train_index], label.iloc[validation_index]
        X_validation = X_validation.reset_index(drop=True)
        y_validation = y_validation.reset_index(drop=True)
        modeldata= X_train
        modellabel = y_train

        modeldata = np.array(modeldata)
        modellabel = np.array(modellabel)
        valdata = np.array(X_validation)
        vallabel = np.array(y_validation)
        clf3 = xgb.XGBClassifier(eval_metric=['logloss', 'auc', 'error'],
                                 use_label_encoder=False,
                                 gamma=0,
                                 learning_rate=0.1,
                                 colsample_bytree=0.8,
                                 max_depth=2,
                                 min_child_weight=1,
                                 n_estimators=20,
                                 seed=1440,
                                 reg_lambda=0.1
                                 )

        clf1 = SVC(kernel='rbf', probability=True, gamma=0.001, C=3)
        clf2 = RandomForestClassifier(n_jobs=-1,
                                      random_state=255,
                                      n_estimators=20,
                                      max_depth=2,
                                      min_samples_split=2,
                                      min_samples_leaf=2)
        lr = LogisticRegression(random_state=0, multi_class='ovr',
                                penalty='l2', C=100)

        eclf = StackingCVClassifier(n_jobs=-1,random_state=2,
                        classifiers=[clf1,clf2,clf3],
                        meta_classifier=lr,
                        use_probas=False,cv=a)
        eclf.fit(modeldata,modellabel)
        valpred = eclf.predict(valdata)
        accuracy, precision_micro, precision_weighted, recall_micro, recall_weighted,  fl_micro, fl_weighted = performance(vallabel, valpred)
        #print(vallabel, valpred)
        Accuracy.append(accuracy)

        Precision_micro.append(precision_micro)
        Precision_weighted.append(precision_weighted)

        Recall_micro.append(recall_micro)
        Recall_weighted.append(recall_weighted)

        Fl_micro.append(fl_micro)
        Fl_weighted.append(fl_weighted)

        # plot figure

        y_score = eclf.predict_proba(valdata)
        
        y_test = label_binarize(vallabel, classes=[0, 1, 2])
        
#        temp=-1
#        y_test = np.zeros((len(vallabel),2))
#        for i in vallabel:
#            temp=temp+1
#            if i==1:
#               y_test[temp][1]=1
        
        #print(y_score)
        #print(vallabel)
        print(y_test)
        fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])
        # interp:��ֵ �ѽ����ӵ�tprs�б���
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # ����auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # ��ͼ��ֻ��Ҫplt.plot(fpr,tpr),����roc_aucֻ�Ǽ�¼auc��ֵ��ͨ��auc()�����������
        plt.rcParams['font.sans-serif'] = 'Arial'  # ����ȫ�����壬�ᱻ�ֲ����嶥��
        plt.rcParams['axes.unicode_minus'] = False  # ����������ʾ����
        print(" {} fold ".format(m))
        m=m+1
        plt.plot(fpr, tpr, linestyle='--',lw=2,alpha=.6, label='ROC fold %d(area=%0.2f)' % (m, roc_auc))
        
        
        
        
    # ���Խ���
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Luck', alpha=.6)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # ����ƽ��AUCֵ
    std_auc = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc,linestyle='--', lw=2, alpha=.5)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_tpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize=14)
    plt.ylabel('True Positive Rate',fontsize=14)
    plt.title('ROC curves of MP2',fontsize=16)
    plt.legend(loc='lower right')
    plt.savefig('MP2_ROC.png', dpi=500, bbox_inches='tight', transparent=False)
    #plt.show()
    
    
    
    
    Accuracy5 = np.mean(Accuracy)

    Precision_micro5 = np.mean(Precision_micro)
    Precision_weighted5 = np.mean(Precision_weighted)

    Recall_micro5 = np.mean(Recall_micro)
    Recall_weighted5 = np.mean(Recall_weighted)

    Fl_micro5 = np.mean(Fl_micro)
    Fl_weighted5 = np.mean(Fl_weighted)

    return Accuracy5, \
           Precision_micro5, Precision_weighted5, \
           Recall_micro5, Recall_weighted5, \
           Fl_micro5, Fl_weighted5

if __name__ == "__main__":

    whole_data = pd.read_csv('modeldata178_13_label3.csv')
    data,label = dataSplit(whole_data)
    traindata = np.array(whole_data.drop(columns=['label']))
    trainlabel = np.array(whole_data['label'])

    test_data = pd.read_csv('test1013_label3.csv')
    data1 = test_data.drop(columns=['label'])
    label1 = test_data['label']
    testdata = np.array(test_data.drop(columns=['label']))
    testlabel = np.array(test_data['label'])

    ACC_train, PRE1, PRE2, RAC1, RAC2, F0, F1 = model_evaluation(data, label)
    print('training')
    print(ACC_train, PRE1, PRE2, RAC1, RAC2, F0, F1)

    clf3 = xgb.XGBClassifier(eval_metric=['logloss', 'auc', 'error'],
                             use_label_encoder=False,
                             gamma=0,
                             learning_rate=0.1,
                             colsample_bytree=0.8,
                             max_depth=2,
                             min_child_weight=1,
                             n_estimators=20,
                             seed=1440,
                             reg_lambda=0.1
                             )

    clf1 = SVC(kernel='rbf',probability=True,gamma=0.001, C=3)
    clf2 = RandomForestClassifier(n_jobs = -1,
                                    random_state=255,
                                    n_estimators=20,
                                    max_depth=2,
                                    min_samples_split=2,
                                    min_samples_leaf=2)
    lr = LogisticRegression(random_state=0, multi_class='ovr',
                            penalty='l2', C=100)
    eclf = StackingCVClassifier(n_jobs=-1,random_state=2,
                    classifiers=[clf1,clf2,clf3],
                    meta_classifier=lr,
                    use_probas=False,cv=a)
    eclf.fit(traindata,trainlabel)
    # output model
    joblib.dump(eclf, "model2.pkl")

    y_pred = eclf.predict(testdata)
    print(testlabel)
    print(y_pred )

    acc1=accuracy_score(testlabel, y_pred)
    print(acc1)
    proba = eclf.predict_proba(testdata)
    pre_prob = []
    for i in range(len(proba)):
        pre_prob.append(proba[i][1])
        accuracy, precision_micro, precision_weighted,  recall_micro, recall_weighted, fl_micro, fl_weighted = performance(testlabel, y_pred)

    print('The evaluation of test set:')
    print(accuracy, precision_micro, precision_weighted,  recall_micro, recall_weighted,  fl_micro, fl_weighted)

