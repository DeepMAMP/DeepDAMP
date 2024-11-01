import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.callbacks import LearningRateScheduler
import tensorflow.python.keras.backend as K

def lr_schedule(epoch):
    if epoch < 1000:
        return 0.1
    elif 1000 <= epoch < 1500:
        return 0.1
    else:
        return 0.01

plt.style.use('seaborn')
np.random.seed(100)

# 读取数据
# dataset = pd.read_csv('test.csv')
# att_data = pd.read_csv('test.csv')
dataset = pd.read_csv('22-mb.csv')
att_data = pd.read_csv('22-mb.csv')
df = dataset

# 取出目标值
y_t = df.iloc[:, -1]
att_t = pd.DataFrame(y_t.values[0:189])

# 取出用于Attention的数据
att_data = att_data[['LSTM', 'lgbm']].values
# att_data = att_data[['lstm', 'lgbm']].values
# 设置神经网络参数
station_id = 1035  # 选择的监测站1035
# lstm_units = 64  # lstm神经元个数,默认16
batch_size = 256  # 批训练大小
epoch =1500  # 迭代次数
scale = 1.0  # 归一化参数
save_model = ('lstm-att-%d.h5' % station_id)  # 保存下该模型，便于下次加载

# 把数据处理成lstm接受的输入形式
y = att_t.iloc[:, -1]
data = np.array(att_data) / scale

# # 构建模型
# inputs = Input(shape=(2,))  # 注意这里的shape是根据att_data的列数而定的
# attention = Dense(2, activation='softmax', name='attention_vec')(inputs)  # 求解Attention权重
# merged = Multiply(name='attention_multiply')([inputs, attention])  # attention与数据对应数值相乘
# outputs = Dense(1, activation='relu')(merged)
# model = Model(inputs=inputs, outputs=outputs)
# model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])



# Add Dropout layer
inputs = Input(shape=(2,))
attention = Dense(2, activation='softmax', name='attention_vec')(inputs)
merged = Multiply(name='attention_multiply')([inputs, attention])
merged = Dropout(0.3)(merged)  # Add dropout with 20% probability
outputs = Dense(1, activation='relu')(merged)

# Compile model with a custom learning rate
custom_optimizer = adam_v2.Adam(learning_rate=0.0001)  # Adjust the learning rate as needed
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer=custom_optimizer, metrics=['accuracy'])


# 训练模型
# from tensorflow.python.keras.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# history = model.fit(data, y, epochs=epoch, batch_size=batch_size, shuffle=False, validation_split=0.2, callbacks=[early_stopping])

# lr_scheduler = LearningRateScheduler(lr_schedule)
# history = model.fit(data, y, epochs=epoch, batch_size=batch_size, shuffle=False, callbacks=[lr_scheduler])
history = model.fit(data, y, epochs=epoch, batch_size=batch_size, shuffle=False)

# 预测并输出结果
y_pred = model.predict(data) * scale
y_pred_plt = pd.DataFrame(y_pred)
y_pred_plt.to_csv('fusion_resultmb-22.csv', index=False, header=False)
y_pred = (y_pred_plt > 0.5).astype(int)
# Create confusion matrix
conf_matrix = confusion_matrix(y, y_pred)

# Calculate metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

# Plot confusion matrix with metrics
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Values')
plt.ylabel('True Values')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
plt.show()

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Multiply, Input
# # from tensorflow.python.keras.optimizers import Adam
# from tensorflow.python.keras.callbacks import EarlyStopping, LearningRateScheduler
# import matplotlib.pyplot as plt
#
# # 读取数据
# dataset = pd.read_csv('2722.csv')
# att_data = dataset[['LSTM', 'lgbm']].values
# y_t = dataset.iloc[:, -1]
#
# # 设置神经网络参数
# station_id = 1035
# lstm_units = 16
# batch_size = 64
# epoch = 1300
# scale = 1.0
# save_model = ('lstm-att-%d.h5' % station_id)
#
# # 把数据处理成lstm接受的输入形式
# y = y_t.values[0:50]
# data = np.array(att_data) / scale
#
# # Train-test split
# X_train, X_val, y_train, y_val = train_test_split(data, y, test_size=0.2, random_state=42)
#
# # Feature scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
#
# # Build model
# model = Sequential()
# model.add(Dense(32, activation='relu', input_shape=(2,)))
# model.add(Dropout(0.2))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='relu'))
#
# model.compile(loss='mse', optimizer=adam_v2.Adam(lr=0.001), metrics=['mae'])
#
# # Define learning rate scheduler
# def lr_schedule(epoch):
#     if epoch < 50:
#         return 0.001
#     elif 50 <= epoch < 100:
#         return 0.0001
#     else:
#         return 0.00001
#
# lr_scheduler = LearningRateScheduler(lr_schedule)
#
# # Define early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#
# # Train model
# history = model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
#                     validation_data=(X_val, y_val), callbacks=[lr_scheduler, early_stopping])
#
# # Evaluate model on validation set
# y_pred_val = model.predict(X_val) * scale
# mse_val = mean_squared_error(y_val, y_pred_val)
# mae_val = mean_absolute_error(y_val, y_pred_val)
# print(f'Validation MSE: {mse_val}, Validation MAE: {mae_val}')
#
# # Save model
# model.save(save_model)
#
# # Plot training history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
