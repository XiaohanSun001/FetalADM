
"""
Created on Fri May 7  00:34:25  2021

@author: xiaohansun
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from mlxtend.classifier import StackingCVClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import seaborn as sns
import shap
import matplotlib.pyplot as plt

a=5
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
def performance(labelArr, predictArr):
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    for i in range(len(labelArr)):
        if labelArr[i] == 1 and predictArr[i] == 1:
            TP += 1.
        if labelArr[i] == 1 and predictArr[i] == 0:
            FN += 1.
        if labelArr[i] == 0 and predictArr[i] == 1:
            FP += 1.
        if labelArr[i] == 0 and predictArr[i] == 0:
            TN += 1.
    ACC = (TP + TN) / (TP + FN + FP + TN)
    SEN = TP / (TP + FN+0.0000001)
    SPE = TN / (FP + TN+0.0000001)
    PRE = TP / (TP + FP+0.0000001)
    F1 = 2 * (PRE * SEN) / (PRE + SEN+0.0000001)
    fz = float(TP * TN - FP * FN+0.0000001)
    fm = float(np.math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)+0.0000001))
    MCC = fz / fm
    return ACC, SEN, SPE, PRE, F1, MCC

def choose_dupldata(prim_data,valid):
    pro_data = pd.concat([prim_data,valid])
    data1 = pro_data.drop_duplicates(keep=False)
    data2 = pro_data.drop_duplicates(keep='first')
    res_data = data1.append(data2).drop_duplicates(keep=False)
    return res_data

def model_evaluation(data,label,prim_data):
    ACC = []
    SEN = []
    SPE = []
    PRE = []
    F1 = []
    MCC = []

    kfold = StratifiedKFold(n_splits = a, shuffle=True, random_state =20)
    for train_index, validation_index in kfold.split(data, label):
        X_train, X_validation = data.iloc[train_index], data.iloc[validation_index]
        y_train, y_validation = label.iloc[train_index], label.iloc[validation_index]
        X_validation = X_validation.reset_index(drop=True)
        y_validation = y_validation.reset_index(drop=True)
        valid = dataMerge(X_validation,y_validation)
        validation_data = choose_dupldata(prim_data,valid)
        validation_data = validation_data.reset_index(drop=True)

        valdata,vallabel = dataSplit(validation_data)
        modeldata = np.array(X_train)
        modellabel = np.array(y_train)
        valdata = np.array(valdata)
        vallabel = np.array(vallabel)

        clf1 = SVC(kernel='rbf',probability=True,gamma=0.029, C=5)
        clf2 = RandomForestClassifier(n_jobs = -1,
                                        random_state=255,
                                        n_estimators=10,
                                        max_depth=4,
                                        min_samples_split=2,
                                        min_samples_leaf=2)
        lr = LogisticRegression(random_state=0,multi_class='ovr',
                       penalty='l2',C=100)
        eclf = StackingCVClassifier(n_jobs=-1,random_state=2,
                        classifiers=[clf1,clf2],
                        meta_classifier=lr,
                        use_probas=False,cv=a)
        eclf.fit(modeldata,modellabel)
        valpred = eclf.predict(valdata)
        acc, sen, spe, pre, f1, mcc = performance(vallabel, valpred)
        pass
        ACC.append(acc)
        SEN.append(sen)
        SPE.append(spe)
        PRE.append(pre)
        F1.append(f1)
        MCC.append(mcc)
        # AUC.append(auc)
    ACC_5 = np.mean(ACC)
    SEN_5 = np.mean(SEN)
    SPE_5 = np.mean(SPE)
    PRE_5 = np.mean(PRE)
    F1_5 = np.mean(F1)
    MCC_5 = np.mean(MCC)
    # AUC_5 = np.mean(AUC)
    return ACC_5,SEN_5,SPE_5,PRE_5,F1_5,MCC_5

if __name__ == "__main__":

    prim_data = pd.read_csv(r'E:\sxh\DNA论文\lw6\mode_train_356\model\feature_selection3\primary216_13_label2.csv')
    whole_data = pd.read_csv(r'E:\sxh\DNA论文\lw6\mode_train_356\model\feature_selection3\modeldata356_13_label2.csv')
    data,label = dataSplit(whole_data)
    traindata = np.array(whole_data.drop(columns=['label']))
    trainlabel = np.array(whole_data['label'])

    test_data = pd.read_csv(r'E:\sxh\DNA论文\lw6\mode_train_356\model\feature_selection3\test54_13_label2.csv')
    data1 = test_data.drop(columns=['label'])
    label1 = test_data['label']
    testdata = np.array(test_data.drop(columns=['label']))
    testlabel = np.array(test_data['label'])

    # ACC_train,SEN_train,SPE_train,PRE_train,F1_train,MCC_train= model_evaluation(data,label,prim_data)
    # print('training:')
    # print(ACC_train,SEN_train,SPE_train,PRE_train,F1_train,MCC_train)

    clf1 = SVC(kernel='rbf',probability=True,gamma=0.029, C=5)
    clf2 = RandomForestClassifier(n_jobs = -1,
                                    random_state=255,
                                    n_estimators=10,
                                    max_depth=4,
                                    min_samples_split=2,
                                    min_samples_leaf=2)
    lr = LogisticRegression(random_state=0, multi_class='ovr',
                            penalty='l2', C=100)
    eclf = StackingCVClassifier(n_jobs=-1,random_state=2,
                    classifiers=[clf1,clf2],
                    meta_classifier=lr,
                    use_probas=False,cv=a)
    eclf.fit(traindata,trainlabel)
    joblib.dump(eclf, "model1.pkl")

    y_pred = eclf.predict(testdata)
    print(testlabel)
    print(y_pred )

    acc1=accuracy_score(testlabel, y_pred)
    print(acc1)
    proba = eclf.predict_proba(testdata)
    pre_prob = []
    for i in range(len(proba)):
        pre_prob.append(proba[i][1])
    ACC_test,SEN_test,SPE_test,PRE_test,F1_test,MCC_test = performance(testlabel, y_pred)
    print('The evaluation of test set:')
    print(ACC_test,SEN_test,SPE_test,PRE_test,F1_test,MCC_test)

#    def predict_wrapper(X):
#        return eclf.predict_proba(X)
#
#    explainer = shap.Explainer(predict_wrapper,testdata)
#    
#    shap_values = explainer(testdata) 
#    
#    #print(testdata)
#    #shap.summary_plot(shap_values, testdata,show=False, plot_size=(10, 10), color=plt.get_cmap("Set2"),plot_type="dot", max_display=None)
#    shap.plots.beeswarm(shap_values[0], show=False)
#    #plt.gcf().set_size_inches(10, 10)
#       
#    
#    plt.xlabel('Mean (SHAP value)')
#    plt.savefig('shap_summary.png')