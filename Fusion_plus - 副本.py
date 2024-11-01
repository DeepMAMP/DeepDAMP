import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Multiply, Dropout
# from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
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

plt.style.use('seaborn')
np.random.seed(100)

# 读取数据
dataset = pd.read_csv('mb-22.csv')
att_data = pd.read_csv('mb-22.csv')
df = dataset

# 取出目标值
y_t = df.iloc[:, -1]
att_t = pd.DataFrame(y_t.values[0:189])

# 取出用于Attention的数据
att_data = att_data[['LSTM', 'lgbm']].values

# 设置神经网络参数
station_id = 1035  # 选择的监测站1035
batch_size = 256  # 批训练大小
epoch = 1500  # 迭代次数
scale = 1.0  # 归一化参数
save_model = ('bp-model-%d.h5' % station_id)  # 保存下该模型，便于下次加载

# 把数据处理成模型接受的输入形式
y = att_t.iloc[:, -1]
data = np.array(att_data) / scale

# 构建模型
inputs = Input(shape=(2,))
hidden_layer = Dense(64, activation='relu')(inputs)
hidden_layer = Dropout(0.3)(hidden_layer)
outputs = Dense(1, activation='relu')(hidden_layer)

# Compile model with a custom learning rate
custom_optimizer = adam_v2.Adam(learning_rate=0.01)  # Adjust the learning rate as needed
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer=custom_optimizer, metrics=['accuracy'])

# 训练模型
history = model.fit(data, y, epochs=epoch, batch_size=batch_size, shuffle=False)

# 预测并输出结果
y_pred = model.predict(data) * scale
y_pred_plt = pd.DataFrame(y_pred)
y_pred_plt.to_csv('bp_model_result_mb-22.csv', index=False, header=False)
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
