
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV,KFold,train_test_split
from sklearn.metrics import make_scorer , accuracy_score
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score
import lightgbm as lgb
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
filename=path+'primary216_14.csv'
all_data = pd.read_csv(filename)

data,label = dataSplit(all_data)
test_data = pd.read_csv(path+'test54_14.csv')

data1,label1 = dataSplit(test_data)
label1=np.array(label1)

#X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8,random_state=1,stratify=label)
#reg = RandomForestClassifier(oob_score=True,
#                        n_estimators=40,max_depth= 6,min_samples_leaf=2,min_samples_split=3)

X_train, y_train=data,label
                                            
reg = xgb.XGBClassifier()

all_classes = np.unique(label)

# 定义分层交叉验证
stratified_kfold = StratifiedKFold(n_splits=2)

best_model = None
best_score = float('-inf')
best_val_performance = None

# 在每个折叠中进行交叉验证
for train_index, val_index in stratified_kfold.split(X_train, y_train):
    # 拆分训练集和验证集
    X_train_fold, X_val_fold = X_train.values[train_index], X_train.values[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # 拟合模型
    reg.fit(X_train_fold, y_train_fold)
    
    # 在验证集上进行预测并计算准确率
    y_val_pred_proba = reg.predict_proba(X_val_fold)
    
    score = -log_loss(y_val_fold, y_val_pred_proba,labels=all_classes)
    
    # 计算验证集上的性能
    y_val_pred = reg.predict(X_val_fold)
    val_performance = performance(y_val_fold, y_val_pred,y_val_pred_proba)    
    
    # 保存最佳模型及其性能
    if score > best_score:
        best_score = score
        best_model = reg
        best_val_performance = val_performance


# 使用整个训练集上的最佳模型进行拟合
if best_model is not None:
	
    grid=best_model.fit(X_train, y_train)
    
    joblib.dump(grid, "model.pkl")      
    y_pred1 = grid.predict(data1)
    y_pred1_proba = grid.predict_proba(data1)    
    acc1 = accuracy_score(label1, y_pred1)
    
    print('acc1')
    print(acc1)
    print(y_pred1)
    print(label1)
    
    per_test = performance(label1, y_pred1,y_pred1_proba)
    print('Best Test Set Performance:')
    print(per_test)
  
    print('Best Model Validation Set Performance:')
    print(val_performance)  
    
    out=open('results.txt','a')
    out.write(f'{filename} \n') 
    out.write('Best Test Set Performance: \n')
    out.write('accuracy, precision_weighted,recall_weighted, fl_weighted,mcc \n')
    #out.write(f'{accuracy}, {precision_micro}, {precision_weighted}, {recall_micro}, {recall_weighted},  {fl_micro}, {fl_weighted} \n')
    out.write(f'{per_test} \n')
 
    out.write('Best Model Validation Set Performance: \n')
    out.write('accuracy, precision_weighted,recall_weighted, fl_weighted,mcc \n')
    out.write(f'{val_performance} \n') 
    out.write(' \n') 
    
    out.close()
    # print('test score : %f' % reg.score(X_test, y_test))
    

else:
    print("No best model found.")