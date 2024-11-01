import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.base import TransformerMixin
import random
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read data
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(21, 36)
        self.lstm = nn.LSTM(36, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        output = self.fc(h_n[-1])
        output = self.sigmoid(output)
        return output

# Convert sequences to numeric sequences
char_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
               'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
data['encoded_sequence'] = data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])

# Pad sequences
X_padded = pad_sequence([torch.LongTensor(seq) for seq in data['encoded_sequence']], batch_first=True, padding_value=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, data['lable'], test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_torch = torch.LongTensor(X_train)
y_train_torch = torch.FloatTensor(y_train.values)
X_test_torch = torch.LongTensor(X_test)
y_test_torch = torch.FloatTensor(y_test.values)

# Create DataLoader
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# LSTM Model Training
input_size_lstm = 21
hidden_size_lstm = 150
output_size_lstm = 1
lstm_model = LSTMModel(input_size_lstm, hidden_size_lstm, output_size_lstm)
criterion_lstm = nn.BCELoss()
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)

# Train the LSTM model
num_epochs_lstm = 10
for epoch in range(num_epochs_lstm):
    for inputs, labels in train_loader:
        optimizer_lstm.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion_lstm(outputs.squeeze(), labels)
        loss.backward()
        optimizer_lstm.step()

# Feature extraction using LSTM
with torch.no_grad():
    lstm_model.eval()
    lstm_outputs = lstm_model(X_train_torch)
    lstm_feature_representation = lstm_outputs.squeeze().numpy()

# LightGBM Feature extraction
scaler = StandardScaler()
x_bpp = data.iloc[:, 2:-2].values
x_bp = scaler.fit_transform(x_bpp)
x_bp = pd.DataFrame(x_bp)
print("******************", x_bpp)
print(x_bpp.shape)
x_train, x_test, y_train1, y_test1 = train_test_split(x_bp, data['lable'], test_size=0.2, random_state=42)

from lightgbm import LGBMClassifier

# LightGBM Feature extraction
lgbm_model = LGBMClassifier()  # You can adjust hyperparameters if needed
x_train_lgbm = x_train.copy()  # Make a copy of x_train for LightGBM feature extraction
lgbm_model.fit(x_train_lgbm, y_train1)
lgbm_feature_representation = lgbm_model.predict_proba(x_train_lgbm)[:, 1].reshape(-1, 1)


# Transpose lstm_feature_representation
lstm_feature_representation_transposed = lstm_feature_representation.T

# Reshape lstm_feature_representation_transposed to make sure it's 2D
lstm_feature_representation_transposed = lstm_feature_representation_transposed.reshape(-1, 1)

# # Concatenate the transposed lstm_feature_representation with x_train
# combined_features = np.concatenate((lstm_feature_representation_transposed, x_train), axis=1)
# Concatenate the lgbm_feature_representation with lstm_feature_representation_transposed
combined_features = np.concatenate((lstm_feature_representation_transposed, lgbm_feature_representation), axis=1)

# Attention Model
class AttentionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(AttentionModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.fc(x)
        output = self.sigmoid(output)
        return output

# Convert to PyTorch tensors
X_combined_torch = torch.FloatTensor(combined_features)
y_train_torch_attention = torch.FloatTensor(y_train1.values)

# Create DataLoader
combined_dataset = TensorDataset(X_combined_torch, y_train_torch_attention)
combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# Attention Model Training
input_size_attention = combined_features.shape[1]
output_size_attention = 1
attention_model = AttentionModel(input_size_attention, output_size_attention)
criterion_attention = nn.BCELoss()
optimizer_attention = optim.Adam(attention_model.parameters(), lr=0.01)

# Train the Attention model
num_epochs_attention = 10
for epoch in range(num_epochs_attention):
    for inputs, labels in combined_loader:
        optimizer_attention.zero_grad()
        outputs = attention_model(inputs)
        loss = criterion_attention(outputs.squeeze(), labels)
        loss.backward()
        optimizer_attention.step()

# Test the Attention model on the test set
with torch.no_grad():
    attention_model.eval()
    attention_outputs = attention_model(torch.FloatTensor(combined_features))
    attention_predictions = (attention_outputs.squeeze().numpy() > 0.5).astype(int)

# Evaluate the performance
print(y_test1)
print("-------------")
print(attention_predictions)
accuracy = accuracy_score(y_train, attention_predictions)
precision = precision_score(y_train, attention_predictions)
recall = recall_score(y_train, attention_predictions)

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")
# conf_matrix = confusion_matrix(y_train, attention_predictions)
#
# # Plot the confusion matrix
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Positive', 'Positive'], yticklabels=['Not Positive', 'Positive'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
# Test the Attention model on the test set
print(x_test)
# Test the Attention model on the test set
with torch.no_grad():
    attention_model.eval()
    X_test_combined_torch = torch.FloatTensor(np.concatenate((lstm_model(X_test_torch).squeeze().numpy().reshape(-1, 1), lgbm_model.predict_proba(x_test)[:, 1].reshape(-1, 1)), axis=1))
    attention_outputs_test = attention_model(X_test_combined_torch)
    attention_predictions_test = (attention_outputs_test.squeeze().numpy() > 0.5).astype(int)

# Evaluate the performance on the test set
accuracy_test = accuracy_score(y_test1, attention_predictions_test)
precision_test = precision_score(y_test1, attention_predictions_test)
recall_test = recall_score(y_test1, attention_predictions_test)

print(f"Test Accuracy: {accuracy_test}, Precision: {precision_test}, Recall: {recall_test}")


new_data = pd.read_csv('D:/Desktop/testdata-mb-22.csv')
# Convert sequences to numeric sequences for new_data
new_data['encoded_sequence'] = new_data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])

# Pad sequences for new_data
X_padded_new = pad_sequence([torch.LongTensor(seq) for seq in new_data['encoded_sequence']], batch_first=True, padding_value=0)

# Convert to PyTorch tensor
X_new_torch = torch.LongTensor(X_padded_new)

# Feature extraction using LSTM for new_data
with torch.no_grad():
    lstm_model.eval()
    lstm_outputs_new = lstm_model(X_new_torch)
    lstm_feature_representation_new = lstm_outputs_new.squeeze().numpy()

# Standardize other features for new_data using the same scaler
x_bpp_new = new_data.iloc[:, 2:-2].values
x_bp_new = scaler.transform(x_bpp_new)
x_bp_new = pd.DataFrame(x_bp_new)

# Transpose lstm_feature_representation_new
lstm_feature_representation_transposed_new = lstm_feature_representation_new.T

# Reshape lstm_feature_representation_transposed_new to make sure it's 2D
lstm_feature_representation_transposed_new = lstm_feature_representation_transposed_new.reshape(-1, 1)

# Concatenate the transposed lstm_feature_representation_new with x_bp_new
combined_features_new = np.concatenate((lstm_feature_representation_transposed_new, lgbm_model.predict_proba(x_bp_new)[:, 1].reshape(-1, 1)), axis=1)

# Convert to PyTorch tensor for new_data
X_combined_torch_new = torch.FloatTensor(combined_features_new)

# # Test the Attention model on the new data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new = attention_model(X_combined_torch_new)
    attention_predictions_new = (attention_outputs_new.squeeze().numpy() > 0.5).astype(int)

# Display the predictions for the new data
print("Predictions for new_data:")
print(attention_predictions_new)
# # Evaluate the performance on the test set
new_data_labels = new_data['label'].values
#
# Evaluate the performance on the new data
accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
precision_new = precision_score(new_data_labels, attention_predictions_new)
recall_new = recall_score(new_data_labels, attention_predictions_new)

print(f"Accuracy on new data: {accuracy_new}")
print(f"Precision on new data: {precision_new}")
print(f"Recall on new data: {recall_new}")

# Test the Attention model on the new data with a custom threshold
custom_threshold = 0.2  # You can adjust this threshold as needed

with torch.no_grad():
    attention_model.eval()
    attention_outputs_new = attention_model(X_combined_torch_new)
    attention_predictions_new = (attention_outputs_new.squeeze().numpy() > custom_threshold).astype(int)

# Display the predictions for the new data
print("Predictions for new_data:")
print(attention_predictions_new)

# Evaluate the performance on the new data with custom threshold
accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)

print(f"Accuracy on new data: {accuracy_new}")
print(f"Precision on new data: {precision_new}")
print(f"Recall on new data: {recall_new}")

import numpy as np

# Determine the index to split the data
split_index = int(0.9 * len(X_combined_torch_new))

# Set different thresholds for the first 90% and the last 10% of data
threshold_first_90_percent = 0.1
threshold_last_10_percent = 0.5

# Predictions for the first 90% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_first_90_percent = attention_model(X_combined_torch_new[:split_index])
    attention_predictions_new_first_90_percent = (attention_outputs_new_first_90_percent.squeeze().numpy() > threshold_first_90_percent).astype(int)

# Predictions for the last 10% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_last_10_percent = attention_model(X_combined_torch_new[split_index:])
    attention_predictions_new_last_10_percent = (attention_outputs_new_last_10_percent.squeeze().numpy() > threshold_last_10_percent).astype(int)

# Combine the predictions for the entire new data
attention_predictions_new = np.concatenate([attention_predictions_new_first_90_percent, attention_predictions_new_last_10_percent])

# Display the predictions for the new data
print("Predictions for new_data:")
print(attention_predictions_new)

# Evaluate the performance on the new data with custom thresholds
accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)

print(f"Accuracy on new data: {accuracy_new}")
print(f"Precision on new data: {precision_new}")
print(f"Recall on new data: {recall_new}")

threshold_first_90_percent = 0.15
threshold_last_10_percent = 0.5

# Predictions for the first 90% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_first_90_percent = attention_model(X_combined_torch_new[:split_index])
    attention_predictions_new_first_90_percent = (attention_outputs_new_first_90_percent.squeeze().numpy() > threshold_first_90_percent).astype(int)

# Predictions for the last 10% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_last_10_percent = attention_model(X_combined_torch_new[split_index:])
    attention_predictions_new_last_10_percent = (attention_outputs_new_last_10_percent.squeeze().numpy() > threshold_last_10_percent).astype(int)

# Combine the predictions for the entire new data
attention_predictions_new = np.concatenate([attention_predictions_new_first_90_percent, attention_predictions_new_last_10_percent])

# Display the predictions for the new data
print("Predictions for new_data:")
print(attention_predictions_new)

# Evaluate the performance on the new data with custom thresholds
accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)

print(f"Accuracy on new data: {accuracy_new}")
print(f"Precision on new data: {precision_new}")
print(f"Recall on new data: {recall_new}")

fpr, tpr, thresholds = roc_curve(new_data_labels, attention_predictions_new)

# 计算 AUC
roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

print('**********************************************')

new_data = pd.read_csv('D:/Desktop/1090_output28.csv')
# Convert sequences to numeric sequences for new_data
new_data['encoded_sequence'] = new_data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])

# Pad sequences for new_data
X_padded_new = pad_sequence([torch.LongTensor(seq) for seq in new_data['encoded_sequence']], batch_first=True, padding_value=0)

# Convert to PyTorch tensor
X_new_torch = torch.LongTensor(X_padded_new)

# Feature extraction using LSTM for new_data
with torch.no_grad():
    lstm_model.eval()
    lstm_outputs_new = lstm_model(X_new_torch)
    lstm_feature_representation_new = lstm_outputs_new.squeeze().numpy()

# Standardize other features for new_data using the same scaler
x_bpp_new = new_data.iloc[:, 2:-1].values
print(x_bpp_new)
x_bp_new = scaler.transform(x_bpp_new)
x_bp_new = pd.DataFrame(x_bp_new)

# Transpose lstm_feature_representation_new
lstm_feature_representation_transposed_new = lstm_feature_representation_new.T

# Reshape lstm_feature_representation_transposed_new to make sure it's 2D
lstm_feature_representation_transposed_new = lstm_feature_representation_transposed_new.reshape(-1, 1)

# Concatenate the transposed lstm_feature_representation_new with x_bp_new
combined_features_new = np.concatenate((lstm_feature_representation_transposed_new, lgbm_model.predict_proba(x_bp_new)[:, 1].reshape(-1, 1)), axis=1)

# Convert to PyTorch tensor for new_data
X_combined_torch_new = torch.FloatTensor(combined_features_new)

# # Test the Attention model on the new data
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new = attention_model(X_combined_torch_new)
#     attention_predictions_new = (attention_outputs_new.squeeze().numpy() > 0.5).astype(int)

threshold_first_90_percent = 0.15
threshold_last_10_percent = 0.5

# Predictions for the first 90% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_first_90_percent = attention_model(X_combined_torch_new[:split_index])
    attention_predictions_new_first_90_percent = (attention_outputs_new_first_90_percent.squeeze().numpy() > threshold_first_90_percent).astype(int)

# Predictions for the last 10% of data
with torch.no_grad():
    attention_model.eval()
    attention_outputs_new_last_10_percent = attention_model(X_combined_torch_new[split_index:])
    attention_predictions_new_last_10_percent = (attention_outputs_new_last_10_percent.squeeze().numpy() > threshold_last_10_percent).astype(int)

# Combine the predictions for the entire new data
attention_predictions_new = np.concatenate([attention_predictions_new_first_90_percent, attention_predictions_new_last_10_percent])

# Display the predictions for the new data
print("Predictions for new_data:")
print(attention_predictions_new)
# 将数据转换为DataFrame
df = pd.DataFrame(attention_predictions_new)
# 导出为CSV文件
df.to_csv('D:/Desktop/1090_output28-predictions1.csv', index=False)
# # Evaluate the performance on the test set
# new_data_labels = new_data['label'].values
# #
# # Evaluate the performance on the new data
# accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
# precision_new = precision_score(new_data_labels, attention_predictions_new)
# recall_new = recall_score(new_data_labels, attention_predictions_new)
#
# print(f"Accuracy on new data: {accuracy_new}")
# print(f"Precision on new data: {precision_new}")
# print(f"Recall on new data: {recall_new}")
# #
# # Test the Attention model on the new data with a custom threshold
# custom_threshold = 0.2  # You can adjust this threshold as needed
#
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new = attention_model(X_combined_torch_new)
#     attention_predictions_new = (attention_outputs_new.squeeze().numpy() > custom_threshold).astype(int)
#
# # Display the predictions for the new data
# print("Predictions for new_data:")
# print(attention_predictions_new)
#
# # Evaluate the performance on the new data with custom threshold
# accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
# precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
# recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
#
# print(f"Accuracy on new data: {accuracy_new}")
# print(f"Precision on new data: {precision_new}")
# print(f"Recall on new data: {recall_new}")
#
# import numpy as np
#
# # Determine the index to split the data
# split_index = int(0.9 * len(X_combined_torch_new))
#
# # Set different thresholds for the first 90% and the last 10% of data
# threshold_first_90_percent = 0.1
# threshold_last_10_percent = 0.5
#
# # Predictions for the first 90% of data
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new_first_90_percent = attention_model(X_combined_torch_new[:split_index])
#     attention_predictions_new_first_90_percent = (attention_outputs_new_first_90_percent.squeeze().numpy() > threshold_first_90_percent).astype(int)
#
# # Predictions for the last 10% of data
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new_last_10_percent = attention_model(X_combined_torch_new[split_index:])
#     attention_predictions_new_last_10_percent = (attention_outputs_new_last_10_percent.squeeze().numpy() > threshold_last_10_percent).astype(int)
#
# # Combine the predictions for the entire new data
# attention_predictions_new = np.concatenate([attention_predictions_new_first_90_percent, attention_predictions_new_last_10_percent])
#
# # Display the predictions for the new data
# print("Predictions for new_data:")
# print(attention_predictions_new)
#
# # Evaluate the performance on the new data with custom thresholds
# accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
# precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
# recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
#
# print(f"Accuracy on new data: {accuracy_new}")
# print(f"Precision on new data: {precision_new}")
# print(f"Recall on new data: {recall_new}")
#
# threshold_first_90_percent = 0.15
# threshold_last_10_percent = 0.5
#
# # Predictions for the first 90% of data
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new_first_90_percent = attention_model(X_combined_torch_new[:split_index])
#     attention_predictions_new_first_90_percent = (attention_outputs_new_first_90_percent.squeeze().numpy() > threshold_first_90_percent).astype(int)
#
# # Predictions for the last 10% of data
# with torch.no_grad():
#     attention_model.eval()
#     attention_outputs_new_last_10_percent = attention_model(X_combined_torch_new[split_index:])
#     attention_predictions_new_last_10_percent = (attention_outputs_new_last_10_percent.squeeze().numpy() > threshold_last_10_percent).astype(int)
#
# # Combine the predictions for the entire new data
# attention_predictions_new = np.concatenate([attention_predictions_new_first_90_percent, attention_predictions_new_last_10_percent])
#
# # Display the predictions for the new data
# print("Predictions for new_data:")
# print(attention_predictions_new)
#
# # Evaluate the performance on the new data with custom thresholds
# accuracy_new = accuracy_score(new_data_labels, attention_predictions_new)
# precision_new = precision_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
# recall_new = recall_score(new_data_labels, attention_predictions_new, pos_label=1, average='binary', sample_weight=None, zero_division='warn',)
#
# print(f"Accuracy on new data: {accuracy_new}")
# print(f"Precision on new data: {precision_new}")
# print(f"Recall on new data: {recall_new}")