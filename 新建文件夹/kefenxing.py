import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns#%matplotlib inline
from sklearn import model_selection
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()
# scaler1 = StandardScaler()
# data=pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-415-513 - mbfea.csv")
data = pd.read_csv("D:/Desktop/allgongkai28.csv")
# data=pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-415-513 - 24fea.csv")
# data=pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-415-513_qufeature.csv")
# data=pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-415-513_qufeatures.csv")
# data=pd.read_csv("D:/Desktop/苦味肽/quan640.csv")
feature_names = data.columns[2:-1]
print(feature_names)
data.head(10)
x_bpp = data.iloc[0:2462, 2:-1].values
# x_bpp = data.drop(columns=['ID', 'Sequences', 'lable'])[0:415]
x_bp=scaler.fit(x_bpp)
x_bp=scaler.transform(x_bpp)
# x_bp = scaler1.fit_transform(x_bp)
x_bp=pd.DataFrame(x_bp)
y_bp = data.iloc[0:2462, -1].values
# y_bp = data['lable'][0:415]
print(x_bp.shape)
x_nbpp = data.iloc[2462:, 2:-1].values
# x_nbpp = data.drop(columns=['ID', 'Sequences', 'lable'])[415:]
x_nbp=scaler.fit(x_nbpp)
x_nbp=scaler.transform(x_nbpp)
# x_nbp = scaler1.fit_transform(x_nbp)
x_nbp=pd.DataFrame(x_nbp)
y_nbp = data.iloc[2462:, -1].values
# y_nbp = data['lable'][415:]
random_state = random.randint(0, 10000000)
# random_state = 92844
x_bp_train, x_bp_test, y_bp_train, y_bp_test = train_test_split(x_bp, y_bp, test_size=0.2, random_state=random_state)
x_nbp_train, x_nbp_test, y_nbp_train, y_nbp_test = train_test_split(x_nbp, y_nbp, test_size=0.2, random_state=random_state)
print(" Random State:", random_state)
# Combining the training and testing sets of bitter and non-bitter peptide data
x_train = np.concatenate((x_bp_train, x_nbp_train), axis=0)
print(len(x_train))
print(x_train)
y_train = np.concatenate((y_bp_train, y_nbp_train), axis=0)
x_test = np.concatenate((x_bp_test, x_nbp_test), axis=0)
y_test = np.concatenate((y_bp_test, y_nbp_test), axis=0)

# df_x = pd.DataFrame(np.concatenate((x_bp_test, x_nbp_test), axis=0))
# df_y = pd.DataFrame(np.concatenate((y_bp_test, y_nbp_test), axis=0))
#
# # 将 DataFrame 导出到 Excel
# with pd.ExcelWriter('output.xlsx') as writer:
#     df_x.to_excel(writer, sheet_name='X_Data', index=False)
#     df_y.to_excel(writer, sheet_name='Y_Data', index=False)

def fit (x,y,params=None):
    folds= KFold(n_splits=10,shuffle=True,random_state=6)
    accuracy=0
    clf=lgb.LGBMClassifier(**params)
    for fold_,(trn_idx,val_idx) in enumerate(folds.split(data)):
        clf.fit(x.iloc[trn_idx],y[trn_idx])
        pred=clf.predict(x.iloc[val_idx])
        accuracy+=accuracy_score(y[val_idx],pred)/folds.n_splits
    accuracy
    return accuracy

def grid (x,y,params):
    default= {'n_job':-1}
    init=fit(x,y,default,)
    for key in params.keys():
        for i in params[key]:
            dic=default.copy()
            dic[key]=i
            score=fit(x,y,dic)
            print (dic,score)
            if score>init:
                init=score
                default=dic
    print (default,init)
# param={'max_depth':[3,4,5,6,7],'num_leaves':[4,8,12,16],'subsample':[0.5,0.8,1],'colsample_bytree':[0.8,0.9],
#        'reg_alpha':[1,2,3,4,10]}
# grid(train_x,train_y,param)

# cross_val_accuracy = fit(x_train, y_train, default)
# print("Cross-Validation Accuracy:", cross_val_accuracy)
#模型训练
import graphviz
default={ 'n_job': -1,'subsample' : 0.5}
clf=lgb.LGBMClassifier(**default)
clf.fit(x_train,y_train)

print("0000",clf)
#绘制决策树
lgb.create_tree_digraph(clf)
#特征重要性
lgb.plot_importance(clf)
plt.show()

# 获取特征重要性
feature_importance = clf.feature_importances_
print(feature_importance)
# 获取非零重要性的特征索引
non_zero_importance_indices = np.where(feature_importance > 0)[0]
print(non_zero_importance_indices)
# 获取非零重要性的特征名
non_zero_feature_names = [feature_names[i] for i in non_zero_importance_indices]
print(non_zero_feature_names)
# 获取非零重要性的特征重要性
non_zero_feature_importance = [feature_importance[i] for i in non_zero_importance_indices]
print(non_zero_feature_importance)
# 根据特征重要性的顺序对特征名和重要性进行排序
sorted_indices = np.argsort(non_zero_feature_importance)[::-1]  # 降序排序的索引
print(sorted_indices)
sorted_feature_names = [non_zero_feature_names[i] for i in sorted_indices]
print(sorted_feature_names)
sorted_feature_importance = [non_zero_feature_importance[i] for i in sorted_indices]
print(sorted_feature_importance)

# 绘制特征重要性图，只包含非零重要性的特征
plt.figure(figsize=(10, 6))
plt.barh(sorted_feature_names, sorted_feature_importance,  height=0.5)
plt.xlabel('Feature Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # 反转Y轴，使特征名称按原始顺序显示
# 显示每个特征的具体重要性值
for i, v in enumerate(sorted_feature_importance):
    plt.text(v+2, i, f'{v:}', color='black', va='center')

# 添加网格线
plt.grid(axis='x', linestyle='-', alpha=0.6)
plt.grid(axis='y', linestyle='-', alpha=0.6)
# 显示图
plt.show()

pre=clf.predict(x_test)
pre
pre=pd.DataFrame(pre)
pre.to_csv("D:/Desktop/抗菌肽/数据/testdata-result.csv",index=False)


#预测
# realpredata = pd.read_csv("D:/pythonProject/code/APIN_Ubuntu/安全域问题/testdata-mb-23-15.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata23-15.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-mb -23+15 - fea.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-mb -23+15_23fea.csv")
realpredata = pd.read_csv("D:/Desktop/testdata-mb-22.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata-mb -23+15_fea24.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata23-15_qufeature.csv")
# realpredata = pd.read_csv("D:/Desktop/抗菌肽/数据/testdata23-15_qufeatures.csv")
realpredata1 = realpredata.iloc[:, 2:-1].values
realresult = realpredata.iloc[:, -1].values
print("预测特征",realpredata1)
realdata = scaler.fit(realpredata1)
realdata = scaler.transform(realpredata1)
# realdata = scaler1.fit_transform(realdata)
realdata = pd.DataFrame(realdata)
print("未知",realdata)
realpre = realdata.values
result = clf.predict(realpre)
result = pd.DataFrame(result,columns=["result"])
print("预测结果",result)
result.to_csv("D:/Desktop/抗菌肽/数据/testdata-mb-22-result.csv",index=False)
count_0 = result[result["result"] == 0].shape[0]
count_1 = result[result["result"] == 1].shape[0]
print("0 的个数:", count_0)
print("1 的个数:", count_1)





from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(y_test, pre)
print(confusion_mat)

def plot_confusion_matrix(confusion_mat):
    '''''将混淆矩阵画图并显示出来'''
    plt.matshow(confusion_mat, cmap=plt.cm.Reds)
    plt.title('Confusion matrix')
    plt.colorbar()
    for i in range (len(confusion_mat)):
        for j in range (len (confusion_mat)):
            plt.annotate(confusion_mat[j,i],xy=(i,j),horizontalalignment='center', verticalalignment='center')
#     tick_marks = np.arange(confusion_mat.shape[0])
#     plt.xticks(tick_marks, tick_marks)
#     plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plot_confusion_matrix(confusion_mat)

from sklearn.metrics import confusion_matrix

confusion_mat = confusion_matrix(y_test, pre)

# Extracting TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = confusion_mat.ravel()
# Calculating ACC
ACC = (TP + TN) / (TP + TN + FP + FN)
# Calculating PRE
PRE = TP / (TP + FP)
# Calculating SN
SN = TP / (TP + FN)
# Calculating F1
F1 = 2 * (PRE * SN) / (PRE + SN)
# Calculating MCC
MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5

# Printing the metrics
print("Accuracy (ACC):", ACC)
print("Precision (PRE):", PRE)
print("Sensitivity (SN):", SN)
print("F1 Score:", F1)
print("Matthews correlation coefficient (MCC):", MCC)

#**************************************************************************
confusion_mat = confusion_matrix(realresult, result)
print(confusion_mat)


def plot_confusion_matrix(confusion_mat):
    '''''将混淆矩阵画图并显示出来'''
    plt.matshow(confusion_mat, cmap=plt.cm.Reds)
    plt.title('Confusion matrix')
    plt.colorbar()
    for i in range(len(confusion_mat)):
        for j in range(len(confusion_mat)):
            plt.annotate(confusion_mat[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    #     tick_marks = np.arange(confusion_mat.shape[0])
    #     plt.xticks(tick_marks, tick_marks)
    #     plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


plot_confusion_matrix(confusion_mat)

from sklearn.metrics import confusion_matrix, roc_auc_score

confusion_mat = confusion_matrix(realresult, result)
# Extracting TP, TN, FP, FN from the confusion matrix
TN, FP, FN, TP = confusion_mat.ravel()
# Calculating ACC
ACC = (TP + TN) / (TP + TN + FP + FN)
# Calculating PRE
PRE = TP / (TP + FP)
# Calculating SN
SN = TP / (TP + FN)
# Calculating F1
F1 = 2 * (PRE * SN) / (PRE + SN)
# Calculating MCC
MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
# Calculating AUC
# AUC = roc_auc_score(y_test, pre)

# Printing the metrics
print("Accuracy (ACC):", ACC)
print("Precision (PRE):", PRE)
print("Sensitivity (SN):", SN)
print("F1 Score:", F1)
print("Matthews correlation coefficient (MCC):", MCC)
# print("Area Under the Curve (AUC):", AUC)


# ss = pd.read_csv("D:/Desktop/allgongkai28.csv")
# s = ss.iloc[:, 2:-1].values
# #可分性
# scaler = MinMaxScaler()
# # g1=scaler.fit(x_train)
# # g1=scaler.transform(x_train)
# g1=scaler.fit(s)
# g1=scaler.transform(s)
# g1=pd.DataFrame(g1)
# print("训练集归一化结果",g1)
#
#
# out_mean = []
#
# for col in range(g1.shape[1]):
#     out = []
#     for i in range(2462):
#         a = g1.iloc[i:i+1, col].values
#         for j in range(2533):
#             b = g1.iloc[2462+j:2462+j+1, col].values
#             dist = np.linalg.norm(a - b)
#             out.append(dist)
#
# # for col in range(g1.shape[1]):
# #     out = []
# #     for i in range(332):
# #         a = g1.iloc[i:i+1, col].values
# #         for j in range(410):
# #             b = g1.iloc[332+j:332+j+1, col].values
# #             dist = np.linalg.norm(a - b)
# #             out.append(dist)
#
#     out_mean.append(np.average(out))
#
# names = ss.columns[2:-1]
#
# out_mean_df = pd.DataFrame(out_mean, columns=['out_mean'], index=g1.columns)
# print(out_mean_df)
# sorted_df = out_mean_df.sort_values('out_mean')
# print(sorted_df)
# # Create the bar plot
# plt.figure(figsize=(10, 6))
# out_mean_df.plot.bar()
# plt.xlabel('Index')
# plt.ylabel('out_mean')
# plt.xticks(range(len(names)), names, rotation='vertical')
# plt.title('Mean Distance for each Column')
# plt.show()