
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
from sklearn.tree  import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import warnings
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data
    
filename= r'E:\sxh\DNA论文\lw6\mode_train_356\model\feature_selection3\modeldata356_13_label2.csv'
all_data = pd.read_csv(filename)

random_state=3
shuffled_data = all_data.sample(frac=1, random_state=2)

data,label = dataSplit(shuffled_data)



# data1,label1 = dataSplit(t18_data)
# data2,label2 = dataSplit(t13_data)

X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8,random_state=1,stratify=label)
# X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, random_state=1,stratify=y_trainval)
import sys

original_stdout = sys.stdout

with open('SVMoutput.txt', 'a') as f:
    sys.stdout = f
    print(filename)
    print(Counter(y_test),Counter(y_train))
    
    regressor = SVC(random_state=5,kernel= 'rbf')
    parameters = {'gamma':np.arange(1e-3,0.03,0.001),'C':range(3,15,1)}
    scores= ['precision','recall']
    
    print("# Tuning hyper-parameters for accurary" )
    print()
    
    # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
    clf1 = GridSearchCV(SVC(), parameters, cv=10,
                           scoring='accuracy')
    # 用训练集训练这个学习器 clf
    clf1.fit(X_train, y_train)
    
     # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
    print(clf1.best_params_)
    
    print("Detailed classification report:")
    
    y_true, y_pred = y_test, clf1.predict(X_test)
        
    print(classification_report(y_true, y_pred))
    print()

sys.stdout = original_stdout
#print()
#means = clf1.cv_results_['mean_test_score']
#stds = clf1.cv_results_['std_test_score']

# 看一下具体的参数间不同数值的组合后得到的分数是多少
# for mean, std, params in zip(means, stds, clf1.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))

