
# import tensorflow.compat.v2 as tf
#
#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.layers import Embedding, LSTM, Dense
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.losses import kullback_leibler_divergence

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# 假设你有一个包含序列和标签的数据集，其中1表示抗菌肽，0表示非抗菌肽
# 请确保你的数据集已经做了适当的处理，例如将序列转换为数值特征

# 读取数据集，假设数据集有两列：'sequence'和'label'
# sequence列包含抗菌肽或非抗菌肽的序列，label列包含标签（1或0）
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, matthews_corrcoef
# Define PyTorch LSTM model
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

# Read data
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Convert sequences to numeric sequences
# char_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
#                'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'Z': 0
#                , 'X': 0, 'U': 0, 'B': 0, 'J': 0, 'O': 0}
char_to_int = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11,
               'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, }
data['encoded_sequence'] = data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])

# Pad sequences
X_padded = pad_sequence([torch.LongTensor(seq) for seq in data['encoded_sequence']], batch_first=True, padding_value=0)
print(X_padded)

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

# Define model and optimizer
input_size = 21
hidden_size = 150
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    model.eval()
    outputs = model(X_test_torch)
    predictions = (outputs.squeeze() > 0.5).float()

    # 将 Tensor 转换为 NumPy 数组
    outputs_array = outputs.numpy()
    # 将 NumPy 数组转换为 DataFrame
    df = pd.DataFrame(outputs_array)
    # 将 DataFrame 保存为 CSV 文件
    df.to_csv('lstm_test1.csv', index=False, header=False)

    accuracy = (predictions == y_test_torch).float().mean()
    print(f"Test Accuracy: {accuracy.item()}")
    # Calculate additional metrics
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"MCC: {mcc}")
    print("__________________________")

# Save the trained model
torch.save(model.state_dict(), 'trained_model.pth')

# Load the test data
test_data = pd.read_csv('D:/Desktop/testdata-mb-22.csv')

# Convert sequences to numeric sequences
test_data['encoded_sequence'] = test_data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])

# Pad sequences
X_test_padded = pad_sequence([torch.LongTensor(seq) for seq in test_data['encoded_sequence']], batch_first=True, padding_value=0)

# Convert to PyTorch tensor
X_test_torch = torch.LongTensor(X_test_padded)

# Create DataLoader
test_dataset = TensorDataset(X_test_torch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Set shuffle to False for test data

# Initialize model and load trained parameters
model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('trained_model.pth'))

# Make predictions on the test data
model.eval()
with torch.no_grad():
    all_out = []
    all_predictions = []
    for inputs in test_loader:
        outputs = model(inputs[0])  # inputs[0] since DataLoader returns a tuple
        predictions = (outputs.squeeze() > 0.5).float()
        all_predictions.extend(predictions.numpy())
        all_out.extend(outputs.numpy())

# Convert predictions to DataFrame
test_results = pd.DataFrame({'Predictions': all_predictions}, index=test_data.index)
all_out = pd.DataFrame(all_out)
all_out.to_csv('lstm_testdata-mb-22.csv', index=False, header=False)
# Calculate accuracy and other metrics
accuracy = accuracy_score(test_data['label'], test_results['Predictions'])
precision = precision_score(test_data['label'], test_results['Predictions'])
recall = recall_score(test_data['label'], test_results['Predictions'])
f1 = f1_score(test_data['label'], test_results['Predictions'])
mcc = matthews_corrcoef(test_data['label'], test_results['Predictions'])

print(f"Test Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")
print("__________________________")
# Confusion Matrix
conf_matrix = confusion_matrix(test_data['label'], test_results['Predictions'])
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# # Load the test data
# test_data = pd.read_csv('D:/Desktop/testdata-mb.csv')
# # test_data = pd.read_csv('D:\Desktop\stage-1.csv')
# # Convert sequences to numeric sequences
# test_data['encoded_sequence'] = test_data['Sequences'].apply(lambda x: [char_to_int[char] for char in x])
#
# # Pad sequences
# X_test_padded = pad_sequence([torch.LongTensor(seq) for seq in test_data['encoded_sequence']], batch_first=True, padding_value=0)
#
# # Convert to PyTorch tensor
# X_test_torch = torch.LongTensor(X_test_padded)
#
# # Create DataLoader
# test_dataset = TensorDataset(X_test_torch)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Set shuffle to False for test data
#
# # Initialize model and load trained parameters
# model = LSTMModel(input_size, hidden_size, output_size)
# model.load_state_dict(torch.load('trained_model.pth'))
#
# # Make predictions on the test data
# model.eval()
# with torch.no_grad():
#     all = []
#     all_predictions = []
#     for inputs in test_loader:
#         outputs = model(inputs[0])  # inputs[0] since DataLoader returns a tuple
#         predictions = (outputs.squeeze() > 0.5).float()
#         all_predictions.extend(predictions.numpy())
#         all.extend(outputs.numpy())
#
# # Convert predictions to DataFrame
# test_results = pd.DataFrame({'Predictions': all_predictions}, index=test_data.index)
# all = pd.DataFrame(all)
# all.to_csv('lstm_mb1.csv', index=False, header=False)
# # Calculate accuracy and other metrics
# accuracy = accuracy_score(test_data['label'], test_results['Predictions'])
# precision = precision_score(test_data['label'], test_results['Predictions'])
# recall = recall_score(test_data['label'], test_results['Predictions'])
# f1 = f1_score(test_data['label'], test_results['Predictions'])
# mcc = matthews_corrcoef(test_data['label'], test_results['Predictions'])
#
# print(f"Test Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print(f"MCC: {mcc}")
#
# # Plot confusion matrix
# conf_matrix = confusion_matrix(test_data['label'], test_results['Predictions'])
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()