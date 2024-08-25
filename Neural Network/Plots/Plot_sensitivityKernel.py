
import os
import numpy as np
import matplotlib.pyplot as plt

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/kernel_size/4layers' # working folder path
os.chdir(project_folder) # changing the path


# Load the feature maps
data = np.load('feature_maps_all55.npz')

# Iterate over each layer to create a separate figure
for i, key in enumerate(sorted(data.files)):  # Sort keys to maintain layer order
    feature_maps = data[key]  # Load the feature maps for this layer
    num_feature_maps = feature_maps.shape[3]  # Number of feature maps in the layer

    # Calculate number of rows (ceil of the number of feature maps divided by 16)
    num_rows = (num_feature_maps + 15) // 16

    # Create a figure for this layer
    fig, axes = plt.subplots(num_rows, 16, figsize=(20, num_rows*2), squeeze=False)
    fig.suptitle(f'Layer {i+1} Feature Maps', fontsize=20)

    for j in range(num_feature_maps):
        row, col = divmod(j, 16)  # Determine the row and column to place the subplot
        ax = axes[row, col]
        im = ax.imshow(feature_maps[0, :, :, j], aspect='auto', cmap='viridis',origin='lower')
        ax.axis('off')
        ax.set_title(f'Map {j+1}', fontsize=10)

    # Hide unused axes in the last row if necessary
    for k in range(j + 1, num_rows * 16):
        row, col = divmod(k, 16)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for the main title
    plt.show()

# Load the feature maps
data = np.load('feature_maps_all33.npz')

# Iterate over each layer to create a separate figure
for i, key in enumerate(sorted(data.files)):  # Sort keys to maintain layer order
    feature_maps = data[key]  # Load the feature maps for this layer
    num_feature_maps = feature_maps.shape[3]  # Number of feature maps in the layer

    # Calculate number of rows (ceil of the number of feature maps divided by 16)
    num_rows = (num_feature_maps + 15) // 16

    # Create a figure for this layer
    fig, axes = plt.subplots(num_rows, 16, figsize=(20, num_rows*2), squeeze=False)
    fig.suptitle(f'Layer {i+1} Feature Maps', fontsize=20)

    for j in range(num_feature_maps):
        row, col = divmod(j, 16)  # Determine the row and column to place the subplot
        ax = axes[row, col]
        im = ax.imshow(feature_maps[0, :, :, j], aspect='auto', cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Map {j+1}', fontsize=10)

    # Hide unused axes in the last row if necessary
    for k in range(j + 1, num_rows * 16):
        row, col = divmod(k, 16)
        axes[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for the main title
    plt.show()

test_loss55 = np.load('test_scores_all55.npy')
test_loss35 = np.load('test_scores_all35.npy')
test_loss33 = np.load('test_scores_all33.npy')

print(test_loss55)
print(test_loss35)
print(test_loss33)


project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/kernel_size/8layers' # working folder path
os.chdir(project_folder) # changing the path

test_loss55 = np.load('test_scores_all55.npy')
test_loss35 = np.load('test_scores_all35.npy')
test_loss33 = np.load('test_scores_all33.npy')

print(test_loss55)
print(test_loss35)
print(test_loss33)