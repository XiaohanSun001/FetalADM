
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
import numpy as np
import warnings
from sklearn.model_selection import ShuffleSplit
warnings.filterwarnings('ignore')



def dataSplit(data):
    X = data.drop(columns=['label'])
    y = data['label']
    return X,y
def dataMerge(X,y):
    merged_data = pd.concat([y.reset_index(drop=True), X], axis=1)
    return merged_data
all_data = pd.read_csv('../../smotedata2/traindata_356_27.csv')
X,Y= dataSplit(all_data)


# #构建RF模型
RFC_ = RFC()                               # 随机森林
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

score = []  # 用于存储交叉验证的得分
best_features = []  # 用于存储最优特征组
best_num_features = 0  # 用于存储最优特征数

# 建立列表
for i in range(1, 28, 1):
    rfe = RFE(RFC_, n_features_to_select=i, step=1)   # 最优特征
    X_wrapper = rfe.fit_transform(X, Y)
    once = cross_val_score(RFC_, X_wrapper, Y, cv=cv).mean()                      # 交叉验证
    score.append(once)

    if once == max(score):  # 如果是当前最大得分，保存这些特征名称
        best_num_features = i  # 更新最优特征数量
        best_features = X.columns[rfe.support_]

# 输出最优的分类结果和对应的特征数量
print("最佳数量和排序")

print(f"最优分类结果: {score}")
print(f"对应的特征数量: {best_num_features}")
print(f"最优特征组的特征名称: {list(best_features)}")

plt.figure(figsize=[10, 8])
major_ticks_x=np.linspace(0, 27, 10)
major_ticks_y=np.linspace(0.80, 0.98, 10)
plt.xticks(major_ticks_x)
plt.yticks(major_ticks_y)
plt.grid(which="major", alpha=0.6)
plt.plot(range(1, 28, 1), score, marker='o', linewidth=2.0)
#plt.xticks(range(1, 21, 1))
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 20,
}

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size' : 10,
}
plt.xlabel('Features',font2)
plt.ylabel('Accuracy(Cross-Validation)',font2)
plt.savefig('RF_RFE_number.png')
#plt.show()

RFC_.fit(X, Y)

# 获取特征的重要性
importances = RFC_.feature_importances_

# 创建特征名称列表
features = X.columns

# 按照重要性排序特征
indices = np.argsort(importances)

# 打印出每个特征及其对应的特征重要性分数
print("Feature Importance Scores:")
for i in indices:
    print(f"{features[i]}: {importances[i]:.4f}")  # 打印特征名称和其对应的重要性分数，保留四位小数


# 绘制特征重要性图
plt.figure(figsize=[10, 8])
#plt.title("Feature Importance")
plt.barh(range(len(features)), importances[indices], align="center")
plt.yticks(range(len(features)), [features[i] for i in indices])

for i in range(len(features)):
    plt.text(importances[indices[i]] + 0.0025, i,  # 数值略微偏移，避免遮挡
             f'{importances[indices[i]]:.3f}',  # 格式化显示数值
             va='center', ha='left', fontsize=font1['size'],
             fontname=font1['family'], fontweight=font1['weight'])

plt.xlabel("Feature importance", font2)
plt.ylabel("Feature names",font2)
plt.grid(True, axis="y")
plt.tight_layout()

# 保存特征重要性图
plt.savefig('feature_importance.png')  # 保存特征重要性图
#plt.show()  # 显示特征重要性图