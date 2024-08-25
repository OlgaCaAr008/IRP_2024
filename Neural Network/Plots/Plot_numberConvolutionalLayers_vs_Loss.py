import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/layers/200_stop' # working folder path
os.chdir(project_folder) # changing the path

mpl.rcParams.update({'font.size': 20})

num_layers = 8  # The number of different layer configurations you have
#best_epochs = [1481-500, 1664-500,1897-500,1278-500,1226-500, 2528-500, 3588-500, 3375-500] # 10,000 stop
best_epochs = [0, 0,146-25,170-25,0, 107-25, 44-25, 67-25] # 200 stop

training_losses = []
validation_losses = []
test_losses = []

i = 1
train_loss = np.load(f'train_loss{i}.npy')
val_loss = np.load(f'val_loss{i}.npy')
test_loss = np.load(f'test_loss{i}.npy')

training_losses.append(train_loss)
validation_losses.append(val_loss)
test_losses.append(test_loss)

# Load the data
for i in range(2, num_layers + 1):

    train_loss = np.load(f'train_loss{i}.npy')
    val_loss = np.load(f'val_loss{i}.npy')
    test_loss = np.load(f'test_loss{i}.npy')


    training_losses.append(train_loss[best_epochs[i-1]])
    validation_losses.append(val_loss[best_epochs[i-1]])
    test_losses.append(test_loss)

# Prepare to plot
layers = list(range(1, num_layers + 1))

plt.figure(figsize=(10, 6))
plt.plot(layers, training_losses, 'o-', label='Training Loss', markersize=8)
plt.plot(layers, validation_losses, 's-', label='Validation Loss', markersize=8)
plt.plot(layers, test_losses, '^-', label='Test Loss', markersize=8)
#set log
plt.yscale('log')
plt.xlabel('Number of Convolutional Layers')
plt.ylabel('Loss')

# Setting the y-axis format to use scientific notation
ax = plt.gca()  # Get the current axis

plt.grid(True)
plt.legend()
plt.show()