
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.metrics import make_scorer , accuracy_score,confusion_matrix
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data

def performance(y_true, y_pred,y_pred_proba):

    
    accuracy = accuracy_score(y_true, y_pred)
    precision_micro = precision_score(y_true, y_pred, average='micro')
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_micro = recall_score(y_true, y_pred, average='micro')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    fl_micro = f1_score(y_true, y_pred, average='micro')
    fl_weighted = f1_score(y_true, y_pred, average='weighted')
    y_true_binary = label_binarize(y_true, classes=range(4))
    
    #auc_micro = roc_auc_score(y_true_binary, y_pred_proba, average='micro')
    #auc_weighted = roc_auc_score(y_true_binary, y_pred_proba, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    
    m=[accuracy, \
          precision_weighted, \
           recall_weighted, \
            fl_weighted,mcc]
    metric=[round(i,3) for i in m] 

    return metric
path='../../feature_selection3/'
#filename=path+'modeldata356_14.csv'


test_data1 = pd.read_csv(path+'T208_14_standard.csv')

data1,label1 = dataSplit(test_data1)
label1=np.array(label1)

test_data2 = pd.read_csv(path+'T135_14_standard.csv')

data2,label2 = dataSplit(test_data2)
label2=np.array(label2)
                                            



modelclass=joblib.load("model.pkl")

y_pred1 = modelclass.predict(data1)
cm1 = confusion_matrix(label1, y_pred1)
print(f'confusion matric: T208')
print(cm1)


y_pred2 = modelclass.predict(data2)
cm2 = confusion_matrix(label2, y_pred2)
print(f'confusion matric: T135')
print(cm2)



out=open('results.txt','a')
out.write(f'{len(data1)} \n')
out.write(f'Ture :{label1} \n')
out.write(f'Pred :{y_pred1} \n')
out.write(f'{cm1} \n')

out.write(f'{len(data2)} \n')
out.write(f'Ture :{label2} \n')
out.write(f'Pred :{y_pred2} \n')
out.write(f'{cm2} \n')


out.close()

    
