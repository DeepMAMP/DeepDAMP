import pandas as pd
import time, datetime
from math import sqrt
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
import csv
from keras.models import *
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('./data/yancheng/yc_data.csv')
# print(dataset)
# del作用是删除到变量到对象的引用和变量名称本身
del dataset['时间']
del dataset['气温℃']
del dataset['湿度%']
del dataset['气压hPa']
del dataset['地面风速m/s']
del dataset['风向°']
del dataset['风速(m/s)']
del dataset['Gust']
df = dataset
print(df)
# close = df['close']
# df.drop(labels=['close'], axis=1,inplace = True)
# df.insert(0, 'close', close)

data_train = df.iloc[100:-470, -1:]
data_test = df.iloc[-470:, -1:]
print(data_train.shape, data_test.shape)
# 数据预处理，进行归一化0,1，最大最小标准化MinMax
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train)

data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
print(data_train)
# LSTM参数设置
output_dim = 1
batch_size = 1000  # 1000
epochs = 100
seq_len = 120
hidden_size = 64  # 64
TIME_STEPS = 120
INPUT_DIM = 1
lstm_units = 16

X_train = np.array([data_train[i: i + seq_len, :] for i in range(data_train.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i: i + seq_len, :] for i in range(data_test.shape[0] - seq_len)])
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
maxy = y_test.max()
miny = y_test.min()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

inputs = Input(shape=(TIME_STEPS, INPUT_DIM))

x = Conv1D(filters=32, kernel_size=1, activation='elu')(inputs)  # , padding = 'same'
x = MaxPooling1D(pool_size=2)(x)
x = Dropout(0.1)(x)

lstm_out = Bidirectional(LSTM(lstm_units, activation='elu'), name='bilstm')(x)

output = Dense(1, activation='elu')(lstm_out)
model = Model(inputs=inputs, outputs=output)

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
model.save('./model/fz_model.h5')

model = load_model('./model/fz_model.h5')
# 需要还原原始数据
y_pred = model.predict(X_test)

data_train = scaler.inverse_transform(data_train)
data_test = scaler.inverse_transform(data_test)
y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
y_raw = np.hstack((y_train, y_test))

# RMSE
print('MSE Train loss:', model.evaluate(X_train, y_train, batch_size=batch_size))
print('MSE Test loss:', model.evaluate(X_test, y_test, batch_size=batch_size))
Rmse = sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', Rmse)

y_pred = scaler.inverse_transform(y_pred)

print(y_pred.shape)

y_pred_plt = pd.DataFrame(y_pred)
y_pred_plt.to_csv('./data/yancheng/9.28_lstm.csv', index=False, header=False)
print(y_pred_plt, type(y_pred_plt))

sns.set_style("whitegrid")
plt.figure(figsize=(20, 15))
y_t = df.iloc[-860:, -1]
plt.plot(y_t.values[10:82], label="true")
plt.plot(y_pred_plt[11:83].values, 'r', label="Prediction")
plt.ylabel('speed(m/s)', fontsize=20)
plt.xlabel('times/(h)', fontsize=20)
# plt.plot(preds2[240:361], label="Predicted")

plt.legend(fontsize='20')
plt.show()

