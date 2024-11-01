import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Calculate lengths of peptide sequences
antimicrobial_lengths = antimicrobial['Sequences'].apply(len)
non_antimicrobial_lengths = non_antimicrobial['Sequences'].apply(len)

# Plotting the distribution of peptide lengths
plt.figure(figsize=(10, 6))

# Plot antimicrobial peptide lengths
plt.hist(antimicrobial_lengths, bins=20, alpha=0.4, color='royalblue', edgecolor='black', label='Antimicrobial Peptides')

# Plot non-antimicrobial peptide lengths
plt.hist(non_antimicrobial_lengths, bins=20, alpha=0.4, color='limegreen', edgecolor='black', label='Non-Antimicrobial Peptides')

plt.title('Distribution of Peptide Segment Lengths', fontsize=16)
plt.xlabel('Peptide Segment Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()  # Adjusts spacing to prevent clipping of labels
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Calculate lengths of peptide sequences
antimicrobial_lengths = antimicrobial['Sequences'].apply(len)
non_antimicrobial_lengths = non_antimicrobial['Sequences'].apply(len)

# Plotting the distribution of peptide lengths
plt.figure(figsize=(10, 6))

# Plot antimicrobial peptide lengths
sns.histplot(antimicrobial_lengths, bins=20, color='royalblue', alpha=0.7, element="step", linestyle='-', linewidth=2, label='Antimicrobial Peptides')

# Plot non-antimicrobial peptide lengths
sns.histplot(non_antimicrobial_lengths, bins=20, color='limegreen', alpha=0.7, element="step", linestyle='-', linewidth=2, label='Non-Antimicrobial Peptides')

plt.title('Distribution of Peptide Segment Lengths', fontsize=16)
plt.xlabel('Peptide Segment Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Calculate lengths of peptide sequences
antimicrobial_lengths = antimicrobial['Sequences'].apply(len)
non_antimicrobial_lengths = non_antimicrobial['Sequences'].apply(len)

# Set style
sns.set_style("whitegrid")

# Morandi Color: Serenity Blue
morandi_color_antimicrobial = "#92A8D1"  # Lighter shade for antimicrobial peptides
morandi_color_non_antimicrobial = "#48516B"  # Darker shade for non-antimicrobial peptides

# Plotting the distribution of peptide lengths
plt.figure(figsize=(10, 6))

# Plot antimicrobial peptide lengths
sns.histplot(antimicrobial_lengths, bins=20, color=morandi_color_antimicrobial, alpha=0.7, kde=True, label='Antimicrobial Peptides')

# Plot non-antimicrobial peptide lengths
sns.histplot(non_antimicrobial_lengths, bins=20, color=morandi_color_non_antimicrobial, alpha=0.7, kde=True, label='Non-Antimicrobial Peptides')

plt.title('Distribution of Peptide Segment Lengths', fontsize=16)
plt.xlabel('Peptide Segment Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print('------------------3D')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Read the CSV file
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Define the range of peptide segment lengths and the number of bins
min_length = min(data['Sequences'].apply(len))
max_length = max(data['Sequences'].apply(len))
num_bins = 20  # Adjust this value as needed

# Calculate the bin edges
bin_edges = np.linspace(min_length, max_length, num_bins + 1)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Calculate the histogram for antimicrobial peptides
antimicrobial_hist, _ = np.histogram(antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Calculate the histogram for non-antimicrobial peptides
non_antimicrobial_hist, _ = np.histogram(non_antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Set up the figure and axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D histogram for antimicrobial peptides
bars1 = ax.bar3d(bin_centers, np.zeros_like(bin_centers), np.zeros_like(bin_centers), 1, 1, antimicrobial_hist, color='r')

# Plot the 3D histogram for non-antimicrobial peptides
bars2 = ax.bar3d(bin_centers, np.ones_like(bin_centers), np.zeros_like(bin_centers), 1, 1, non_antimicrobial_hist, color='b')

# Set labels and title
ax.set_xlabel('Peptide Segment Length', fontsize=12)
ax.set_ylabel('Category', fontsize=12)
ax.set_zlabel('Frequency', fontsize=12)
plt.title('3D Histogram of Peptide Segment Lengths', fontsize=16)

# Create legend
legend_elements = [Line2D([0], [0], color='r', lw=10, label='Antimicrobial Peptides'),
                   Line2D([0], [0], color='b', lw=10, label='Non-Antimicrobial Peptides')]
plt.legend(handles=legend_elements)

plt.show()

print('-----------------3D修改')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Read the CSV file
data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Define the range of peptide segment lengths and the number of bins
min_length = min(data['Sequences'].apply(len))
max_length = max(data['Sequences'].apply(len))
num_bins = 20  # Adjust this value as needed

# Calculate the bin edges
bin_edges = np.linspace(min_length, max_length, num_bins + 1)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Calculate the histogram for antimicrobial peptides
antimicrobial_hist, _ = np.histogram(antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Calculate the histogram for non-antimicrobial peptides
non_antimicrobial_hist, _ = np.histogram(non_antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Set up the figure and axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define a custom colormap for the gradient effect
colors = plt.cm.magma(np.linspace(0, 1, num_bins))  # Using magma colormap for a smooth gradient effect
cmap_name = 'custom_gradient'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=num_bins)

# Plot the 3D histogram for antimicrobial peptides with gradient colors
bars1 = ax.bar3d(bin_centers, np.zeros_like(bin_centers), np.zeros_like(bin_centers), 1, 1, antimicrobial_hist, color=cm(antimicrobial_hist / max(antimicrobial_hist)))

# Plot the 3D histogram for non-antimicrobial peptides with gradient colors
bars2 = ax.bar3d(bin_centers, np.ones_like(bin_centers), np.zeros_like(bin_centers), 1, 1, non_antimicrobial_hist, color=cm(non_antimicrobial_hist / max(non_antimicrobial_hist)))

# Set labels and title
ax.set_xlabel('Peptide Segment Length', fontsize=12)
ax.set_ylabel('Category', fontsize=12, labelpad=25)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(['Antimicrobial Peptides', 'Non-Antimicrobial Peptides'])
ax.set_zlabel('Frequency', fontsize=12)
plt.title('3D Histogram of Peptide Segment Lengths', fontsize=16)

# Remove numerical ticks from the y-axis
ax.yaxis.set_tick_params(size=0)

plt.show()


data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Define the range of peptide segment lengths and the number of bins
min_length = min(data['Sequences'].apply(len))
max_length = max(data['Sequences'].apply(len))
num_bins = 20  # Adjust this value as needed

# Calculate the bin edges
bin_edges = np.linspace(min_length, max_length, num_bins + 1)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Calculate the histogram for antimicrobial peptides
antimicrobial_hist, _ = np.histogram(antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Calculate the histogram for non-antimicrobial peptides
non_antimicrobial_hist, _ = np.histogram(non_antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Set up the figure and axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define a custom colormap for the gradient effect
colors = plt.cm.viridis(np.linspace(0, 1, num_bins))  # Using viridis colormap for a smooth gradient effect
cmap_name = 'custom_gradient'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=num_bins)

# Define the width of the bars along the y-axis
bar_width = 0.00005  # Adjust this value to shorten the width of each bar

# Plot the 3D histogram for antimicrobial peptides with gradient colors
for i in range(num_bins):
    ax.bar3d(bin_centers[i], 0.0003 * np.ones_like(bin_centers[i]), np.zeros_like(bin_centers[i]), 1, bar_width, antimicrobial_hist[i], color=cm(i), edgecolor='none')

# Plot the 3D histogram for non-antimicrobial peptides with gradient colors
for i in range(num_bins):
    ax.bar3d(bin_centers[i], -0.0003 * np.ones_like(bin_centers[i]), np.zeros_like(bin_centers[i]), 1, bar_width, non_antimicrobial_hist[i], color=cm(i), edgecolor='none')

# Set labels and title
ax.set_xlabel('Peptide Segment Length', fontsize=12)
ax.set_ylabel('Category', fontsize=12, labelpad=35)  # Adjust the label padding

# Set y-ticks more centered and adjust labels accordingly
ax.set_yticks([-0.0002, 0.0005])
ax.set_yticklabels(['Non-Antimicrobial', 'Antimicrobial'])

ax.set_zlabel('Frequency', fontsize=12)
plt.title('3D Histogram of Peptide Segment Lengths', fontsize=16)

# Remove numerical ticks from the y-axis
ax.yaxis.set_tick_params(size=0)

plt.show()

data = pd.read_csv("D:/Desktop/allgongkai28.csv")

# Filter data for antimicrobial and non-antimicrobial peptides
antimicrobial = data[data['lable'] == 1]  # Assuming label 1 represents antimicrobial peptides
non_antimicrobial = data[data['lable'] == 0]  # Assuming label 0 represents non-antimicrobial peptides

# Define the range of peptide segment lengths and the number of bins
min_length = min(data['Sequences'].apply(len))
max_length = max(data['Sequences'].apply(len))
num_bins = 20  # Adjust this value as needed

# Calculate the bin edges
bin_edges = np.linspace(min_length, max_length, num_bins + 1)

# Calculate the bin centers
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# Calculate the histogram for antimicrobial peptides
antimicrobial_hist, _ = np.histogram(antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Calculate the histogram for non-antimicrobial peptides
non_antimicrobial_hist, _ = np.histogram(non_antimicrobial['Sequences'].apply(len), bins=bin_edges)

# Set up the figure and axes
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Define a custom colormap for the gradient effect
colors = plt.cm.RdBu(np.linspace(0, 1, num_bins))  # Using RdBu colormap for a gradient from red to blue
cmap_name = 'custom_gradient'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=num_bins)

# Define the width of the bars along the y-axis
bar_width = 0.00005  # Adjust this value to shorten the width of each bar

# Plot the 3D histogram for antimicrobial peptides with gradient colors
for i in range(num_bins):
    ax.bar3d(bin_centers[i], 0.0003 * np.ones_like(bin_centers[i]), np.zeros_like(bin_centers[i]), 1, bar_width, antimicrobial_hist[i], color=cm(i), edgecolor='none')

# Plot the 3D histogram for non-antimicrobial peptides with gradient colors
for i in range(num_bins):
    ax.bar3d(bin_centers[i], -0.0003 * np.ones_like(bin_centers[i]), np.zeros_like(bin_centers[i]), 1, bar_width, non_antimicrobial_hist[i], color=cm(i), edgecolor='none')

# Set labels and title
ax.set_xlabel('Peptide Segment Length', fontsize=12)
ax.set_ylabel('Category', fontsize=12, labelpad=35)  # Adjust the label padding

# Set y-ticks more centered and adjust labels accordingly
ax.set_yticks([-0.0002, 0.0005])
ax.set_yticklabels(['Non-Antimicrobial', 'Antimicrobial'])

ax.set_zlabel('Frequency', fontsize=12)
plt.title('3D Histogram of Peptide Segment Lengths', fontsize=16)

# Remove numerical ticks from the y-axis
ax.yaxis.set_tick_params(size=0)

plt.show()

print('__________________原始未筛选数据')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("D:/Desktop/origin.CSV")

# Filter data for antimicrobial peptides and remove missing values
antimicrobial = data[(data['lable'] == 1) & (data['Sequences'].notnull())]

# Calculate lengths of antimicrobial peptide sequences
antimicrobial_lengths = antimicrobial['Sequences'].apply(lambda x: len(str(x)))

# Plotting the distribution of antimicrobial peptide lengths
plt.figure(figsize=(10, 6))

# Plot antimicrobial peptide lengths
sns.histplot(antimicrobial_lengths, bins=20, color='royalblue', alpha=0.7, element="step", linestyle='-', linewidth=2, label='Antimicrobial Peptides')

plt.title('Distribution of Antimicrobial Peptide Segment Lengths', fontsize=16)
plt.xlabel('Peptide Segment Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
print("------------------------------------")
data = pd.read_csv("D:/Desktop/origin.CSV")

# Filter data for antimicrobial peptides and remove missing values
antimicrobial = data[(data['lable'] == 1) & (data['Sequences'].notnull())]

# Calculate lengths of antimicrobial peptide sequences
antimicrobial_lengths = antimicrobial['Sequences'].apply(lambda x: len(str(x)))

# Set style
sns.set_style("whitegrid")

# Plotting the distribution of antimicrobial peptide lengths
plt.figure(figsize=(10, 6))

# Plot antimicrobial peptide lengths
sns.histplot(antimicrobial_lengths, bins=20, color='skyblue', alpha=0.8, kde=True, label='Antimicrobial Peptides')

plt.title('Distribution of Antimicrobial Peptide Segment Lengths', fontsize=16)
plt.xlabel('Peptide Segment Length', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()