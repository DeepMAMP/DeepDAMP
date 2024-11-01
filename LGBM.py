import random

import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import seaborn as sns  # %matplotlib inline
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np

# datar = pd.read_csv('./data/yancheng/yc_data.csv')
# # data=datar.head(10)
# # sns.pairplot(datar)
# data = datar.iloc[4000:-350, :]
# test = datar.iloc[-350:, :]
# # x_train = data[['风速(m/s)', '风向°', 'Gust', '气压hPa', '气温℃']].values
# x_train = data[['风速(m/s)', '风向°', 'Gust']].values
# y_train = data['输出功率(MW)'].values
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


n_folds = 5


def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=15).get_n_splits(data.values)
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)


lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.5))
score = rmsle_cv(lasso)
print("Hist Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
HistGB = HistGradientBoostingRegressor(learning_rate=0.18, max_iter=100, max_leaf_nodes=4)
#
score = rmsle_cv(HistGB)
KNR = KNeighborsRegressor(n_neighbors=8, weights='uniform', algorithm='auto', leaf_size=100, p=1, metric='minkowski')

score1 = rmsle_cv(KNR)
model_xgb1 = xgb.XGBRegressor(learning_rate=2e-2, max_depth=8,
                              min_child_weight=1.1, n_estimators=180,
                              reg_alpha=0.3, reg_lambda=0.7,
                              # subsample=0.5213, silent=1,
                              nthread=-1)
model_xgb1.fit(x_train, y_train)
preds = model_xgb1.predict(x_train)
score2 = mean_squared_error(y_train, preds)

# model_xgb1.predict(np.array([5, 10.0, 0.23,80,10]).reshape(1, -1))
# model_xgb1.predict(np.array([5, 110.0, 0.23]).reshape(1, -1))
print('x_train.shape:', x_train.shape)
# 控制learning_rate=0.2参数变化，可调节预测精度
model_lgb1 = lgb.LGBMRegressor(objective='regression_l2', num_leaves=4,
                               learning_rate=0.2, n_estimators=40,
                               # max_bin = 255, bagging_fraction = 0.8,
                               # bagging_freq = 5, feature_fraction = 0.8,
                               # feature_fraction_seed=9, bagging_seed=9,
                               reg_alpha=0.3, reg_lambda=0.7,
                               # min_data_in_leaf =3, min_sum_hessian_in_leaf = 2
                               )



# 回归参数解释
model_lgb1.fit(x_train, y_train)
preds = model_lgb1.predict(x_train)
score4 = mean_squared_error(y_train, preds)
score4 ** 0.5
r2 = r2_score(y_train, preds)
KNR.fit(x_train, y_train)
# # 创建SVM模型
# for k in ['linear', 'poly', 'rbf', 'sigmoid']:
#     clf = SVR(kernel=k)
#     clf.fit(x_train, y_train)
#     con = clf.score(x_train, y_train)
#     print('``````````````````````````````')
#     print(k, con)
svm_model = SVR(kernel='linear', C=1.0, gamma=0.5)  # 根据需要调整参数
svm_model.fit(x_train, y_train)

# model_lgb1.predict(np.array([3, 9, 1,1000,15]).reshape(1, -1))
# model_lgb1.predict(np.array([4, 10, 2]).reshape(1, -1))
print('model_xgb1.feature_importances_:', model_xgb1.feature_importances_)
# 选择重要特征
# important_features = X.columns[feature_importance > threshold]

# 训练权重值
print('model_lgb1.feature_importances_:', model_lgb1.feature_importances_)
# 获取LightGBM模型的特征表示
# 获取LightGBM模型的特征表示
lgb_feature_importance = model_lgb1.feature_importances_

# 选择重要特征
important_features_lgb = feature_names[lgb_feature_importance > 0.1]

# 转换为列索引的数组
selected_columns = [feature_names.get_loc(feature) for feature in important_features_lgb]

# 提取重要特征的表示
lgb_feature_representation = x_train[:, selected_columns]
print(lgb_feature_representation)


print("-----------------")
# x_val = test[['风速(m/s)', '风向°', 'Gust', '气压hPa', '气温℃']].values
# x_val = test[['风速(m/s)', '风向°', 'Gust']].values
# y_val = test['输出功率(MW)'].values


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


x_val = x_test
y_val = y_test
print('x_val.shape:', x_val.shape)
preds1 = model_xgb1.predict(x_val)
print('preds1.shape:', preds1.shape)
preds2 = model_lgb1.predict(x_val)
print('preds2.shape', preds2.shape)
# preds3 = svm_model.predict(x_val)


# 字典中的key值即为csv中列名
dataframe = pd.DataFrame({'lgbm': preds2})
dataframe['lgbm'] = (dataframe['lgbm'] > 0.5).astype(int)
# print(dataframe)
# dataframe = pd.DataFrame({'KNN': preds3})
# 将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("lgbm_output.csv", index=False, sep=',')
# dataframe.to_csv("./data/yancheng/knn.csv", index=False, sep=',')
# 绘制结果
# plt.figure(figsize=(20, 15))
# plt.title('XGB')
# plt.plot(y_val[0:120], label="True")
# plt.plot(preds1[0:120], label="Predicted")
# plt.xlabel("Number of hours")
# plt.ylabel("Power generated by system (kW)")
# # plt.legend(figsize=15)
# plt.show()

plt.figure(figsize=(20, 15))
sns.set_style("whitegrid")
plt.title('LGB')
plt.plot(y_val[10:72], label="True", )
plt.plot(preds2[10:72], label="Predicted")
plt.xlabel("Number of hours", fontsize=14)
plt.ylabel("Wind Speed(m/s)", fontsize=14)
plt.legend(fontsize=14)
plt.show()
