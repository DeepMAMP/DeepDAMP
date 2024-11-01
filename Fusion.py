# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import seaborn as sns
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import adam_v2

import tensorflow.python.keras.backend as K

plt.style.use('seaborn')
np.random.seed(100)

# dataset = pd.read_csv('./data/yancheng/yc_data.csv')
# att_data = pd.read_csv('./data/yancheng/attention9.28.csv')
dataset = pd.read_csv('2722.csv')
att_data = pd.read_csv('2722.csv')
df = dataset

y_t = df.iloc[-50:, -1]
print(y_t)

att_t = pd.DataFrame(y_t.values[0:50])
att_data = att_data.iloc[0:50, :]
att_data = att_data[['LSTM', 'lgbm']].values
# print(att_data)
# 设置神经网络参数
station_id = 1035  # 选择的监测站1035
lstm_units = 16  # lstm神经元个数,默认16
batch_size = 64  # 批训练大小
epoch = 1300  # 迭代次数
test_ratio = 0.2  # 测试集比例
windows = 4  # 时间窗 4
scale = 1.0  # 归一化参数
save_model = ('lstm-att-%d.h5' % station_id)  # 保存下该模型，便于下次加载
# 读取数据
# 把数据处理成lstm接受的输入形式
y = att_t.iloc[:, -1]

# y=y_train
# y=true
data = np.array(att_data) / scale
print(data)
# cut = round(test_ratio* data.shape[0])
cut = 4
amount_of_features = data.shape[1]
lstm_input = []
data_temp = data
for i in range(len(data_temp) - windows):
    lstm_input.append(data_temp[i:i + windows, :])
lstm_input = np.array(lstm_input)
# print(lstm_input.shape)
lstm_output = y[:-windows]
lstm_output = np.array(lstm_output)
# print(lstm_output.shape)
x_train, y_train, x_test, y_test = lstm_input[:-cut, :, :], lstm_output[:-cut:], lstm_input[-cut:, :, :], lstm_output[
                                                                                                          -cut:]
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 建立模型 ---注意力机制
inputs = Input(shape=(windows, amount_of_features))
lstm_inputs = Permute([2, 1])(inputs)
lstm = LSTM(lstm_units, activation='relu', return_sequences=True)(lstm_inputs)
lstm = Permute([2, 1])(lstm)  # 置换维度
lstm = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(lstm)
attention = Dense(amount_of_features, activation='sigmoid', name='attention_vec')(lstm)  # 求解Attention权重
attention = Activation('softmax', name='attention_weight')(attention)
model = Multiply()([lstm, attention])  # attention与LSTM对应数值相乘
outputs = Dense(1, activation='relu')(model)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()  # 展示模型结构

# 训练模型
history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size,
                    shuffle=False, validation_split=0.2)  # 训练模型epoch次
model.save(save_model)  # 保存模型

# 加载模型
from tensorflow.python.keras.models import load_model

model = load_model(save_model)
sns.set_style("whitegrid")
# 获得网络权重
# weights = np.array(model.get_weights())
# print(weights)
# 输出attention层权重
attention_layer_model = Model(inputs=model.input, outputs=model.get_layer('attention_weight').output)
attention_weight = attention_layer_model.predict(x_train)
attention_weight_final = np.mean(np.array(attention_weight), axis=0)
pd.DataFrame(attention_weight_final, columns=['attention (%)']).plot(kind='bar',
                                                                     title='Attention Mechanism as '
                                                                           'a function of input'
                                                                           ' dimensions.')
plt.show()
# 在训练集上的拟合结果
y_train_predict1 = model.predict(x_train) * scale
y_pred_plt = pd.DataFrame(y_train_predict1)
y_pred_plt.to_csv('fusion_2722.csv', index=False, header=False)

# 模型进行检验
sns.set_style("whitegrid")
plt.figure(figsize=(20, 15))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def up_down_accuracy(y_true, y_pred):
    y_var_test = y_true[1:] - y_true[:len(y_true) - 1]  # 实际涨跌
    y_var_predict = y_pred[1:] - y_pred[:len(y_pred) - 1]  # 原始涨跌
    txt = np.zeros(len(y_var_test))
    for i in range(len(y_var_test - 1)):  # 计算数量
        txt[i] = np.sign(y_var_test[i]) == np.sign(y_var_predict[i])
    result = sum(txt) / len(txt)
    return result


# 在训练集上的拟合结果
