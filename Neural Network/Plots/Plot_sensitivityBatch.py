
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib as mpl

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/batch_size/4layers' # working folder path
os.chdir(project_folder) # changing the path

stop = [824-500, 1240-500, 1701-500, 3327-500, 5425-500, 6199-500]

# Load training history data
history_data16 = np.load('training_history_data4_16.npz')
loss16 = history_data16['loss']
val_loss16 = history_data16['val_loss']

history_data32 = np.load('training_history_data4_32.npz')
loss32 = history_data32['loss']
val_loss32 = history_data32['val_loss']

history_data64 = np.load('training_history_data4_64.npz')
loss64 = history_data64['loss']
val_loss64 = history_data64['val_loss']

history_data128 = np.load('training_history_data4_128.npz')
loss128 = history_data128['loss']
val_loss128 = history_data128['val_loss']

history_data256 = np.load('training_history_data4_256.npz')
loss256 = history_data256['loss']
val_loss256 = history_data256['val_loss']

history_data512 = np.load('training_history_data4_512.npz')
loss512 = history_data512['loss']
val_loss512 = history_data512['val_loss']

test_scores16 = np.load('test_scores4_16.npy')
test_scores32 = np.load('test_scores4_32.npy')
test_scores64 = np.load('test_scores4_64.npy')
test_scores128 = np.load('test_scores4_128.npy')
test_scores256 = np.load('test_scores4_266.npy')
test_scores512 = np.load('test_scores4_512.npy')

mpl.rcParams.update({'font.size': 20})

# Define batch sizes
batch_sizes = [16, 32, 64, 128, 256, 512]

# Average the losses for the last few epochs to stabilize the values
avg_loss = [loss16[stop[0]-1],loss32[stop[1]-1], loss64[stop[2]-1], loss128[stop[3]-1], loss256[stop[4]-1], loss512[stop[5]-1]]

avg_val_loss = [val_loss16[stop[0]-1], val_loss32[stop[1]-1], val_loss64[stop[2]-1],val_loss128[stop[3]-1], val_loss256[stop[4]-1], val_loss512[stop[5]-1]]

avg_test_loss = [test_scores16[0], test_scores32[0], test_scores64[0],test_scores128[0], test_scores256[0], test_scores512[0]]

### Step 2: Plot Data
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, avg_loss, 'o-', label='Training Loss', markersize=8)
plt.plot(batch_sizes, avg_val_loss, 's-', label='Validation Loss', markersize=8)
plt.plot(batch_sizes, avg_test_loss, '^-', label='Test Loss', markersize=8)

plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.xscale('log')  # Optional: Log scale for x-axis if preferred for better visibility
plt.xticks(batch_sizes, labels=batch_sizes)  # Ensure batch sizes are used as labels

# Setting the y-axis format to use scientific notation
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # Enable scientific notation

plt.grid(True)
plt.legend()
plt.show()

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/batch_size/8layers' # working folder path
os.chdir(project_folder) # changing the path

stop = [2172-500, 3045-500, 1246-500, 904-500, 1227-500]

# Load training history data
history_data16 = np.load('training_history_data4_16.npz')
loss16 = history_data16['loss']
val_loss16 = history_data16['val_loss']

history_data32 = np.load('training_history_data4_32.npz')
loss32 = history_data32['loss']
val_loss32 = history_data32['val_loss']

history_data64 = np.load('training_history_data4_64.npz')
loss64 = history_data64['loss']
val_loss64 = history_data64['val_loss']

history_data128 = np.load('training_history_data4_128.npz')
loss128 = history_data128['loss']
val_loss128 = history_data128['val_loss']

history_data256 = np.load('training_history_data4_256.npz')
loss256 = history_data256['loss']
val_loss256 = history_data256['val_loss']

test_scores16 = np.load('test_scores4_16.npy')
test_scores32 = np.load('test_scores4_32.npy')
test_scores64 = np.load('test_scores4_64.npy')
test_scores128 = np.load('test_scores4_128.npy')
test_scores256 = np.load('test_scores4_266.npy')

mpl.rcParams.update({'font.size': 20})

# Define batch sizes
batch_sizes = [16, 32, 64, 128, 256]

# Average the losses for the last few epochs to stabilize the values
avg_loss = [loss16[stop[0]-1],loss32[stop[1]-1], loss64[stop[2]-1], loss128[stop[3]-1], loss256[stop[4]-1]]

avg_val_loss = [val_loss16[stop[0]-1], val_loss32[stop[1]-1], val_loss64[stop[2]-1],val_loss128[stop[3]-1], val_loss256[stop[4]-1]]

avg_test_loss = [test_scores16[0], test_scores32[0], test_scores64[0],test_scores128[0], test_scores256[0]]

### Step 2: Plot Data
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, avg_loss, 'o-', label='Training Loss', markersize=8)
plt.plot(batch_sizes, avg_val_loss, 's-', label='Validation Loss', markersize=8)
plt.plot(batch_sizes, avg_test_loss, '^-', label='Test Loss', markersize=8)

plt.xlabel('Batch Size')
plt.ylabel('Loss')
plt.xscale('log')
plt.yscale('log')
plt.xticks(batch_sizes, labels=batch_sizes)  # Ensure batch sizes are used as labels

# Setting the y-axis format to use scientific notation
ax = plt.gca()  # Get the current axis
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))  # Enable scientific notation

plt.grid(True)
plt.legend()
plt.show()