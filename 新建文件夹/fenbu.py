import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()

# Load data
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Scaling features
x_bpp = data.iloc[0:2462, 2:-1].values
x_bp = scaler.fit_transform(x_bpp)
x_bp = pd.DataFrame(x_bp)

x_nbpp = data.iloc[2462:, 2:-1].values
x_nbp = scaler.fit_transform(x_nbpp)
x_nbp = pd.DataFrame(x_nbp)

# Labels
y_bp = data.iloc[0:2462, -1].values
y_nbp = data.iloc[2462:, -1].values

# Splitting data
random_state = np.random.randint(0, 10000000)
x_bp_train, x_bp_test, y_bp_train, y_bp_test = train_test_split(x_bp, y_bp, test_size=0.2, random_state=random_state)
x_nbp_train, x_nbp_test, y_nbp_train, y_nbp_test = train_test_split(x_nbp, y_nbp, test_size=0.2, random_state=random_state)

# Combining training sets
x_train = np.concatenate((x_bp_train, x_nbp_train), axis=0)
y_train = np.concatenate((y_bp_train, y_nbp_train), axis=0)

# Extract peptide sequences
sequences = data['Sequences']
x_train_sequences = sequences.iloc[:len(x_bp_train) + len(x_nbp_train)]

# Calculate sequence lengths for ABPs and non-ABPs
abp_lengths = [len(seq) for seq, label in zip(x_train_sequences, y_train) if label == 1]
non_abp_lengths = [len(seq) for seq, label in zip(x_train_sequences, y_train) if label == 0]

# Plot distributions
plt.figure(figsize=(10, 6))
plt.hist(abp_lengths, bins=50, color='blue', alpha=0.5, label='Antibacterial Peptides')
plt.hist(non_abp_lengths, bins=50, color='green', alpha=0.5, label='Non-Antibacterial Peptides')
plt.title('Distribution of Peptide Sequence Lengths in Training Set')
plt.xlabel('Peptide Sequence Length')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()
