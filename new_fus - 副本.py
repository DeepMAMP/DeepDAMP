# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.nn.utils.rnn import pad_sequence
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import HistGradientBoostingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import KFold, cross_val_score
# import lightgbm as lgb
# import xgboost as xgb
# from sklearn.svm import SVR
# from sklearn.base import TransformerMixin
# import random
# from sklearn.linear_model import Lasso
# from sklearn.preprocessing import RobustScaler
# from sklearn.metrics import r2_score
#
# # Set seed for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
#
# # Read data
# data = pd.read_csv("D:/Desktop/allgongkai28.csv")
#
# # LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(21, 36)
#         self.lstm = nn.LSTM(36, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.embedding(x)
#         _, (h_n, _) = self.lstm(x)
#         output = self.fc(h_n[-1])
#         output = self.sigmoid(output)
#         return output
#
# # Convert sequences to numeric sequences
# char_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
#                'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
# data['encoded_sequence'] = data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])
#
# # Pad sequences
# X_padded = pad_sequence([torch.LongTensor(seq) for seq in data['encoded_sequence']], batch_first=True, padding_value=0)
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_padded, data['lable'], test_size=0.2, random_state=42)
# # print(y_test)
# # Convert to PyTorch tensors
# X_train_torch = torch.LongTensor(X_train)
# y_train_torch = torch.FloatTensor(y_train.values)
# X_test_torch = torch.LongTensor(X_test)
# y_test_torch = torch.FloatTensor(y_test.values)
#
# # Create DataLoader
# train_dataset = TensorDataset(X_train_torch, y_train_torch)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # LSTM Model Training
# input_size_lstm = 21
# hidden_size_lstm = 150
# output_size_lstm = 1
# lstm_model = LSTMModel(input_size_lstm, hidden_size_lstm, output_size_lstm)
# criterion_lstm = nn.BCELoss()
# optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
#
# # Train the LSTM model
# num_epochs_lstm = 10
# for epoch in range(num_epochs_lstm):
#     for inputs, labels in train_loader:
#         optimizer_lstm.zero_grad()
#         outputs = lstm_model(inputs)
#         loss = criterion_lstm(outputs.squeeze(), labels)
#         loss.backward()
#         optimizer_lstm.step()
#
# # Feature extraction using LSTM
# with torch.no_grad():
#     lstm_model.eval()
#     lstm_outputs = lstm_model(X_train_torch)
#     lstm_feature_representation = lstm_outputs.squeeze().numpy()
#
#
# print(lstm_feature_representation)
# print(len(lstm_feature_representation))
# # LightGBM Feature extraction
# scaler = StandardScaler()
# x_bpp = data.iloc[:, 2:-1].values
# x_bp = scaler.fit_transform(x_bpp)
# x_bp = pd.DataFrame(x_bp)
# x_train, x_test, y_train1, y_test1 = train_test_split(x_bp, data['lable'], test_size=0.2, random_state=42)
# print(len(x_train))
# # print(x_train)
# # print(y_test1)
# # ... (similar preprocessing for x_nbp)
# # Transpose lstm_feature_representation
# # Transpose lstm_feature_representation
# lstm_feature_representation_transposed = lstm_feature_representation.T
#
# # Reshape lstm_feature_representation_transposed to make sure it's 2D
# lstm_feature_representation_transposed = lstm_feature_representation_transposed.reshape(-1, 1)
#
# # Concatenate the transposed lstm_feature_representation with x_train
# combined_features = np.concatenate((lstm_feature_representation_transposed, x_train), axis=1)
# print(combined_features)
#
#
# # Combine features
# # combined_features = np.concatenate((lstm_feature_representation, x_train), axis=1)
#
# # Attention Model
#
# class AttentionModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(AttentionModel, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         attn_weights = self.softmax(x)
#         output = torch.matmul(x.t(), attn_weights).squeeze()  # Adjust the matrix multiplication
#         output = self.fc(output)
#         return output
#
# # Combine LSTM and LGBM features
# attention_input_size = lstm_feature_representation_transposed.shape[1] + x_train.shape[1]
# attention_output_size = 1
# attention_model = AttentionModel(attention_input_size, attention_output_size)
# criterion_attention = nn.BCELoss()
# optimizer_attention = optim.Adam(attention_model.parameters(), lr=0.01)
#
# # Convert combined features to PyTorch tensor
# combined_features_torch = torch.FloatTensor(combined_features)
# y_test_torch = torch.FloatTensor(y_train.values)
#
# # Attention Model Training
# num_epochs_attention = 10
# for epoch in range(num_epochs_attention):
#     optimizer_attention.zero_grad()
#     outputs_attention = attention_model(combined_features_torch)
#     loss_attention = criterion_attention(outputs_attention.squeeze(), y_test_torch)
#     loss_attention.backward()
#     optimizer_attention.step()
#
#
# # Evaluate the Attention Model
# # Evaluate the Attention Model
# with torch.no_grad():
#     attention_model.eval()
#     outputs_attention = attention_model(combined_features_torch)
#     predictions_attention = (outputs_attention.squeeze() > 0.5).float()
#
#     # Ensure y_test_torch and predictions_attention have the same size
#     y_test_torch = y_test_torch[:len(predictions_attention)]
#
#     # Calculate the binary cross-entropy loss
#     loss_attention = criterion_attention(outputs_attention.squeeze(), y_test_torch)
#
#     accuracy_attention = accuracy_score(y_test[:len(predictions_attention)], predictions_attention)
#     precision_attention = precision_score(y_test[:len(predictions_attention)], predictions_attention)
#     recall_attention = recall_score(y_test[:len(predictions_attention)], predictions_attention)
#     f1_attention = f1_score(y_test[:len(predictions_attention)], predictions_attention)
#     mcc_attention = matthews_corrcoef(y_test[:len(predictions_attention)], predictions_attention)
#
#     print(f"Attention Model Accuracy: {accuracy_attention}")
#     print(f"Attention Model Precision: {precision_attention}")
#     print(f"Attention Model Recall: {recall_attention}")
#     print(f"Attention Model F1 Score: {f1_attention}")
#     print(f"Attention Model MCC: {mcc_attention}")
#     print(f"Attention Model Loss: {loss_attention.item()}")
# import random
#
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from torch.nn.utils.rnn import pad_sequence
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
# from sklearn.preprocessing import StandardScaler
# import lightgbm as lgb
#
# # Set seed for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
#
# # Read data
# data = pd.read_csv("D:/Desktop/allgongkai28.csv")
#
# # LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(LSTMModel, self).__init__()
#         self.embedding = nn.Embedding(21, 36)
#         self.lstm = nn.LSTM(36, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.embedding(x)
#         _, (h_n, _) = self.lstm(x)
#         output = self.fc(h_n[-1])
#         output = self.sigmoid(output)
#         return output
#
# # Convert sequences to numeric sequences
# char_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
#                'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
# data['encoded_sequence'] = data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])
#
# # Pad sequences
# X_padded = pad_sequence([torch.LongTensor(seq) for seq in data['encoded_sequence']], batch_first=True, padding_value=0)
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_padded, data['lable'], test_size=0.2, random_state=42)
#
# # Convert to PyTorch tensors
# X_train_torch = torch.LongTensor(X_train)
# y_train_torch = torch.FloatTensor(y_train.values)
# X_test_torch = torch.LongTensor(X_test)
# y_test_torch = torch.FloatTensor(y_test.values)
#
# # Create DataLoader
# train_dataset = TensorDataset(X_train_torch, y_train_torch)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#
# # LSTM Model Training
# input_size_lstm = 21
# hidden_size_lstm = 150
# output_size_lstm = 1
# lstm_model = LSTMModel(input_size_lstm, hidden_size_lstm, output_size_lstm)
# criterion_lstm = nn.BCELoss()
# optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=0.01)
#
# # Train the LSTM model
# num_epochs_lstm = 10
# for epoch in range(num_epochs_lstm):
#     for inputs, labels in train_loader:
#         optimizer_lstm.zero_grad()
#         outputs = lstm_model(inputs)
#         loss = criterion_lstm(outputs.squeeze(), labels)
#         loss.backward()
#         optimizer_lstm.step()
#
# # Feature extraction using LSTM
# with torch.no_grad():
#     lstm_model.eval()
#     lstm_outputs = lstm_model(X_train_torch)
#     lstm_feature_representation = lstm_outputs.squeeze().numpy()
#
# # LightGBM Feature extraction
# scaler = StandardScaler()
# x_bpp = data.iloc[:, 2:-1].values
# x_bp = scaler.fit_transform(x_bpp)
# x_bp = pd.DataFrame(x_bp)
# x_train, x_test, y_train1, y_test1 = train_test_split(x_bp, data['lable'], test_size=0.2, random_state=42)
#
# # Transpose lstm_feature_representation
# lstm_feature_representation_transposed = lstm_feature_representation.T
#
# # Reshape lstm_feature_representation_transposed to make sure it's 2D
# lstm_feature_representation_transposed = lstm_feature_representation_transposed.reshape(-1, 1)
#
# # Concatenate the transposed lstm_feature_representation with x_train
# combined_features = np.concatenate((lstm_feature_representation_transposed, x_train), axis=1)
#
# # Attention Model
# class AttentionModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(AttentionModel, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         attn_weights = self.softmax(x)
#         output = torch.matmul(x.t(), attn_weights).squeeze()
#         output = self.fc(output)
#         return output
#
# # Combine LSTM and LGBM features
# attention_input_size = lstm_feature_representation_transposed.shape[1] + x_train.shape[1]
# attention_output_size = 1
# attention_model = AttentionModel(attention_input_size, attention_output_size)
# criterion_attention = nn.BCELoss()
# optimizer_attention = optim.Adam(attention_model.parameters(), lr=0.01)
#
# # Convert combined features to PyTorch tensor
# combined_features_torch = torch.FloatTensor(combined_features)
# y_test_torch = torch.FloatTensor(y_test.values)
#
# # Attention Model Training
# num_epochs_attention = 10
# for epoch in range(num_epochs_attention):
#     optimizer_attention.zero_grad()
#     outputs_attention = attention_model(combined_features_torch)
#
#     # Modify the target size to match the output size of the attention model
#     y_test_torch_expanded = y_test_torch.view(-1, 1).expand_as(outputs_attention)
#
#     loss_attention = criterion_attention(outputs_attention.squeeze(), y_test_torch_expanded)
#     loss_attention.backward()
#     optimizer_attention.step()
#
# # Evaluate the Attention Model on Test Set
# with torch.no_grad():
#     attention_model.eval()
#     outputs_attention_test = attention_model(combined_features_torch)
#     predictions_attention_test = (outputs_attention_test.squeeze() > 0.5).float()
#
#     # Ensure y_test_torch and predictions_attention_test have the same size
#     y_test_torch = y_test_torch[:len(predictions_attention_test)]
#
#     # Calculate the binary cross-entropy loss
#     loss_attention_test = criterion_attention(outputs_attention_test.squeeze(), y_test_torch)
#
#     accuracy_attention_test = accuracy_score(y_test[:len(predictions_attention_test)], predictions_attention_test)
#     precision_attention_test = precision_score(y_test[:len(predictions_attention_test)], predictions_attention_test)
#     recall_attention_test = recall_score(y_test[:len(predictions_attention_test)], predictions_attention_test)
#     f1_attention_test = f1_score(y_test[:len(predictions_attention_test)], predictions_attention_test)
#     mcc_attention_test = matthews_corrcoef(y_test[:len(predictions_attention_test)], predictions_attention_test)
#
#     print(f"Attention Model Accuracy: {accuracy_attention_test}")
#     print(f"Attention Model Precision: {precision_attention_test}")
#     print(f"Attention Model Recall: {recall_attention_test}")
#     print(f"Attention Model F1 Score: {f1_attention_test}")
#     print(f"Attention Model MCC: {mcc_attention_test}")
#     print(f"Attention Model Loss: {loss_attention_test.item()}")
import random
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
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

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
x_bpp = data.iloc[:, 2:-1].values
x_bp = scaler.fit_transform(x_bpp)
x_bp = pd.DataFrame(x_bp)
x_train, x_test, y_train1, y_test1 = train_test_split(x_bp, data['lable'], test_size=0.2, random_state=42)

# Transpose lstm_feature_representation
lstm_feature_representation_transposed = lstm_feature_representation.T

# Reshape lstm_feature_representation_transposed to make sure it's 2D
lstm_feature_representation_transposed = lstm_feature_representation_transposed.reshape(-1, 1)
print(lstm_feature_representation_transposed.shape)
print(x_train.shape)
# Concatenate the transposed lstm_feature_representation with x_train
combined_features = np.concatenate((lstm_feature_representation_transposed, x_train), axis=1)
print(combined_features)
print(combined_features.shape)
# Attention Model
# class AttentionModel(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(AttentionModel, self).__init__()
#         self.fc = nn.Linear(input_size, output_size)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         attn_weights = self.sigmoid(self.fc(x))  # Apply sigmoid activation
#         output = torch.matmul(x.t(), attn_weights).squeeze()
#         return output
#
# # Combine LSTM and LGBM features
# attention_input_size = lstm_feature_representation_transposed.shape[1] + x_train.shape[1]
# attention_output_size = 1
# attention_model = AttentionModel(attention_input_size, attention_output_size)
# criterion_attention = nn.BCELoss()
# optimizer_attention = optim.Adam(attention_model.parameters(), lr=0.01)

# Attention Model
# BP (Feedforward) Model
# BP Model
class BPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Modify the input_size_bp to match the output size of the Attention model
input_size_bp = attention_output_size  # Modify this line to match the output size of the Attention model
hidden_size_bp = 100
output_size_bp = 1
bp_model = BPModel(input_size_bp, hidden_size_bp, output_size_bp)
criterion_bp = nn.BCELoss()
optimizer_bp = optim.Adam(bp_model.parameters(), lr=0.01)

# Convert combined features to PyTorch tensor
combined_features_torch = torch.FloatTensor(combined_features)

# Modify the target size to match the output size of the BP model
y_train_torch_expanded = y_train_torch.view(-1, 1).expand_as(combined_features_torch)

# BP Model Training
num_epochs_bp = 10
for epoch in range(num_epochs_bp):
    optimizer_bp.zero_grad()
    outputs_bp = bp_model(combined_features_torch)

    # Apply sigmoid activation
    outputs_bp_sigmoid = torch.sigmoid(outputs_bp)

    # Modify the target size to match the output size of the BP model
    y_train_torch_expanded = y_train_torch.view(-1, 1)[:outputs_bp.size(0), :].squeeze()

    loss_bp = criterion_bp(outputs_bp_sigmoid.squeeze(), y_train_torch_expanded)
    loss_bp.backward()
    optimizer_bp.step()

# Evaluate the BP Model on Test Set
with torch.no_grad():
    bp_model.eval()
    outputs_bp_test = bp_model(combined_features_torch)
    predictions_bp_test = (torch.sigmoid(outputs_bp_test.squeeze()) > 0.5).float()

    # Ensure y_test_torch and predictions_bp_test have the same size
    y_test_torch = y_test_torch[:len(predictions_bp_test)]

    # Calculate the binary cross-entropy loss
    loss_bp_test = criterion_bp(torch.sigmoid(outputs_bp_test.squeeze()), y_test_torch)

    accuracy_bp_test = accuracy_score(y_test[:len(predictions_bp_test)], predictions_bp_test)
    precision_bp_test = precision_score(y_test[:len(predictions_bp_test)], predictions_bp_test)
    recall_bp_test = recall_score(y_test[:len(predictions_bp_test)], predictions_bp_test)
    f1_bp_test = f1_score(y_test[:len(predictions_bp_test)], predictions_bp_test)
    mcc_bp_test = matthews_corrcoef(y_test[:len(predictions_bp_test)], predictions_bp_test)

    # Confusion matrix
    confusion_matrix_bp = confusion_matrix(y_test[:len(predictions_bp_test)], predictions_bp_test)

    print(f"BP Model Accuracy: {accuracy_bp_test}")
    print(f"BP Model Precision: {precision_bp_test}")
    print(f"BP Model Recall: {recall_bp_test}")
    print(f"BP Model F1 Score: {f1_bp_test}")
    print(f"BP Model MCC: {mcc_bp_test}")
    print(f"BP Model Loss: {loss_bp_test.item()}")
    print("Confusion Matrix:")
    print(confusion_matrix_bp)




# # Load the new dataset
# new_data = pd.read_csv('D:/Desktop/testdata-mb-22.csv')
#
# # Convert sequences to numeric sequences (similar to training data)
# new_data['encoded_sequence'] = new_data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])
#
# # Pad sequences
# X_new_padded = pad_sequence([torch.LongTensor(seq) for seq in new_data['encoded_sequence']], batch_first=True, padding_value=0)
#
# # Convert to PyTorch tensor
# X_new_torch = torch.LongTensor(X_new_padded)
#
# # Feature extraction using LSTM for the new dataset
# with torch.no_grad():
#     lstm_model.eval()
#     lstm_outputs_new = lstm_model(X_new_torch)
#     lstm_feature_representation_new = lstm_outputs_new.squeeze().numpy()
#
# # LightGBM Feature extraction for the new dataset
# x_bpp_new = new_data.iloc[:, 2:-1].values
# x_bp_new = scaler.transform(x_bpp_new)
# x_bp_new = pd.DataFrame(x_bp_new)
#
# # Transpose lstm_feature_representation_new
# lstm_feature_representation_transposed_new = lstm_feature_representation_new.T
#
# # Reshape lstm_feature_representation_transposed_new to make sure it's 2D
# lstm_feature_representation_transposed_new = lstm_feature_representation_transposed_new.reshape(-1, 1)
#
# # Concatenate the transposed lstm_feature_representation_new with x_bp_new
# combined_features_new = np.concatenate((lstm_feature_representation_transposed_new, x_bp_new), axis=1)
#
# # Convert combined features to PyTorch tensor for the new dataset
# combined_features_torch_new = torch.FloatTensor(combined_features_new)
#
# # Use the trained attention model for prediction on the new dataset
# with torch.no_grad():
#     attention_model.eval()
#     outputs_attention_new = attention_model(combined_features_torch_new)
# predictions_attention_new = (torch.sigmoid(outputs_attention_new.squeeze()) > 0.5).float()
#
# # Display predictions for the new dataset
# print("Predictions for the new dataset:")
# print(predictions_attention_new)
# labels_new = new_data['label']
#
# # Calculate the confusion matrix for the new dataset
# # Ensure predictions and labels have the same number of samples
# predictions_attention_new = predictions_attention_new[:len(labels_new)]
# # Display the lengths of labels_new and predictions_attention_new
# print("Length of labels_new:", len(labels_new))
# print("Length of predictions_attention_new:", len(predictions_attention_new))
#
# # Calculate the confusion matrix
# confusion_matrix_new = confusion_matrix(labels_new, predictions_attention_new)
#
# # Display the confusion matrix for the new dataset
# print("Confusion Matrix for the new dataset:")
# print(confusion_matrix_new)


# ...

# Evaluate the Attention Model on Test Set
# ...

# Evaluate the Attention Model on Test Set
# with torch.no_grad():
#     attention_model.eval()
#     outputs_attention_test = attention_model(combined_features_torch)
#     predictions_attention_test = (torch.sigmoid(outputs_attention_test.squeeze()) > 0.5).float()
#
#     # Ensure y_test_torch and predictions_attention_test have the same size
#     y_test_torch = y_test_torch[:len(predictions_attention_test)]
#
#     # Apply sigmoid activation
#     outputs_attention_test_sigmoid = torch.sigmoid(outputs_attention_test.squeeze())
#
#     # Modify the target size to match the output size of the attention model
#     y_test_torch_expanded = y_test_torch.view(-1, 1).repeat(1, outputs_attention_test_sigmoid.size(0))
#
#     # Ensure y_test_torch_expanded has the same size as outputs_attention_test_sigmoid
#     # Ensure y_test_torch_expanded has the same size as outputs_attention_test_sigmoid
#     # Ensure y_test_torch_expanded has the same size as outputs_attention_test_sigmoid
#     y_test_torch_expanded = y_test_torch_expanded[:outputs_attention_test_sigmoid.size(0)]
#
#     # Expand dimensions of y_test_torch_expanded to match the size of outputs_attention_test_sigmoid
#     y_test_torch_expanded = y_test_torch_expanded.unsqueeze(1).expand(-1, outputs_attention_test_sigmoid.size(1))
#
#     # Continue with the calculation of loss
#     loss_attention_test = criterion_attention(outputs_attention_test_sigmoid.squeeze(), y_test_torch_expanded.squeeze())
#
#     accuracy_attention_test = accuracy_score(y_test_torch, predictions_attention_test)
#     precision_attention_test = precision_score(y_test_torch, predictions_attention_test)
#     recall_attention_test = recall_score(y_test_torch, predictions_attention_test)
#     f1_attention_test = f1_score(y_test_torch, predictions_attention_test)
#     mcc_attention_test = matthews_corrcoef(y_test_torch, predictions_attention_test)
#
#     # Confusion matrix
#     confusion_matrix_attention = confusion_matrix(y_test_torch, predictions_attention_test)
#
#     print(f"Attention Model Accuracy: {accuracy_attention_test}")
#     print(f"Attention Model Precision: {precision_attention_test}")
#     print(f"Attention Model Recall: {recall_attention_test}")
#     print(f"Attention Model F1 Score: {f1_attention_test}")
#     print(f"Attention Model MCC: {mcc_attention_test}")
#     print(f"Attention Model Loss: {loss_attention_test.item()}")
#     print("Confusion Matrix:")
#     print(confusion_matrix_attention)



