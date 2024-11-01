import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn import model_selection
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
ss = pd.read_csv("D:/Desktop/allgongkai28.csv")
s = ss.iloc[:, 2:-1].values
#可分性
scaler = MinMaxScaler()
# g1=scaler.fit(x_train)
# g1=scaler.transform(x_train)
g1=scaler.fit(s)
g1=scaler.transform(s)
g1=pd.DataFrame(g1)
print("训练集归一化结果",g1)


out_mean = []

for col in range(g1.shape[1]):
    out = []
    for i in range(2462):
        a = g1.iloc[i:i+1, col].values
        for j in range(2533):
            b = g1.iloc[2462+j:2462+j+1, col].values
            dist = np.linalg.norm(a - b)
            out.append(dist)

# for col in range(g1.shape[1]):
#     out = []
#     for i in range(332):
#         a = g1.iloc[i:i+1, col].values
#         for j in range(410):
#             b = g1.iloc[332+j:332+j+1, col].values
#             dist = np.linalg.norm(a - b)
#             out.append(dist)

    out_mean.append(np.average(out))

names = ss.columns[2:-1]

out_mean_df = pd.DataFrame(out_mean, columns=['out_mean'], index=g1.columns)
print(out_mean_df)
sorted_df = out_mean_df.sort_values('out_mean')
print(sorted_df)
# Create the bar plot
plt.figure(figsize=(10, 6))
out_mean_df.plot.bar()
# plt.xlabel('Index')
# plt.ylabel('out_mean')
# plt.xticks(range(len(names)), names, rotation='vertical')
# plt.title('Mean Distance for each Column')
plt.xlabel('Index', fontsize=14)
plt.ylabel('out_mean', fontsize=14)
plt.xticks(range(len(names)), names, rotation='vertical', fontsize=12)
plt.title('Mean Distance for each Column', fontsize=16)
plt.show()