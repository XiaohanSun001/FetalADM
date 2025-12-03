
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.metrics import make_scorer , accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.tree  import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import xgboost as xgb
import warnings
from sklearn.neighbors import KNeighborsClassifier as KNN
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data
all_data = pd.read_csv('../modeldata178_13_label3.csv')
data,label = dataSplit(all_data)
# data1,label1 = dataSplit(t18_data)
# data2,label2 = dataSplit(t13_data)
random_state=3
shuffled_data = all_data.sample(frac=1, random_state=1)

data,label = dataSplit(shuffled_data)

X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8,random_state=1,stratify=label)
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1,stratify=y_trainval)

import sys

original_stdout = sys.stdout

with open('XGBoostoutput.txt', 'a') as f:
    sys.stdout = f

    print(Counter(y_test),Counter(y_train))
    
    regressor = xgb.XGBClassifier(random_state=0)
    parameters = {'n_estimators':range(10,100,10),
                  "max_depth": range(2, 8),
                    "min_child_weight": range(0, 11),
                  "colsample_bytree" : np.arange(0.0, 1.0, 0.2)}



    scorin_fnc = make_scorer(accuracy_score)
    kflod = KFold(n_splits=10)
    
    grid = GridSearchCV(regressor, parameters, scoring=scorin_fnc, cv=kflod)
    grid = grid.fit(X_train, y_train)
    reg = grid.best_estimator_
    
    print('best score:%f' % grid.best_score_)

    print('Best parameters: %s' % grid.best_params_)
    # for key in parameters.keys():
    #     print('%s:%d' % (key, reg.get_params()[key]))
    
    print('test score : %f' % reg.score(X_test, y_test))
    
    print()
    
    print("Detailed classification report:")
    
    y_true, y_pred = y_test, reg.predict(X_test)
    
    # 打印在测试集上的预测结果与真实值的分数
    print(classification_report(y_true, y_pred))

sys.stdout = original_stdout


