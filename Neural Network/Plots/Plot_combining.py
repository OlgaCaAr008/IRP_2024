

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import binary_dilation

project_folder = '/content/drive/My Drive/Colab Notebooks' # working folder path
os.chdir(project_folder) # changing the path

geometry = np.load('geo.npy') # n x Ny x Nx

np.random.seed(42)  # For reproducibility
indices = np.arange(geometry.shape[0])
np.random.shuffle(indices)

train_size = int(0.8 * len(indices))
test_indices = indices[train_size:]

Input_test1 = geometry[test_indices]

project_folder = '/content/drive/My Drive/Colab Notebooks/combine' # working folder path
os.chdir(project_folder) # changing the path

predictions = np.load('predictions.npy')
Vx_test = np.load('Vx_test.npy')
Vy_test = np.load('Vy_test.npy')
Rho_test = np.load('Rho_test.npy')
test_scores = np.load('test_scores.npy')

project_folder = '/content/drive/My Drive/Colab Notebooks/combine/av' # working folder path
os.chdir(project_folder) # changing the path

predictions_av = np.load('predictions_av.npy')
test_scores_av = np.load('test_scores_av.npy')

project_folder = '/content/drive/My Drive/Colab Notebooks/combine/w1' # working folder path
os.chdir(project_folder) # changing the path

predictions_w1 = np.load('predictions_w1.npy')
test_scores_w1 = np.load('test_scores_w1.npy')


project_folder = '/content/drive/My Drive/Colab Notebooks/combine/w2' # working folder path
os.chdir(project_folder) # changing the path

predictions_w2 = np.load('predictions_w2.npy')
test_scores_w2 = np.load('test_scores_w2.npy')

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)
def compute_mea(ground_truth,prediction):
  absolute_errors = np.abs(ground_truth - prediction)
  mae = np.mean(absolute_errors)
  return mae

predictions = predictions_w2 # predictions10 predictions100 predictions1000 predictions10000
mae_v = []
mae_rho = []
mre_v = []
mre_rho = []

for j in range(predictions[0].shape[0]):
  predicted_vx = predictions[0][j].reshape(300, 300)
  predicted_vy = predictions[1][j].reshape(300, 300)
  predicted_velocity_magnitude = compute_velocity_magnitude(predicted_vx,predicted_vy)
  original_velocity_magnitude = compute_velocity_magnitude(Vx_test[j,:].reshape(300, 300) , Vy_test[j,:].reshape(300, 300) )
  predicted_rho = predictions[2][j].reshape(300, 300)
  mae_v.append(compute_mea(original_velocity_magnitude,predicted_velocity_magnitude))
  mae_rho.append(compute_mea(Rho_test[j,:].reshape(300, 300),predicted_rho))


  # Compute MRE for velocity and density, ignoring infinities
  mre_v_values = np.abs(original_velocity_magnitude - predicted_velocity_magnitude) / np.abs(original_velocity_magnitude)
  mre_rho_values = np.abs(Rho_test[j, :].reshape(300, 300) - predicted_rho) / np.abs(Rho_test[j, :].reshape(300, 300))
  mre_v_filtered = mre_v_values[np.isfinite(mre_v_values)]
  mre_rho_filtered = mre_rho_values[np.isfinite(mre_rho_values)]

  mre_v.append(np.mean(mre_v_filtered)*100)
  mre_rho.append(np.mean(mre_rho_filtered)*100)

print('MAE_vel:', np.mean(mae_v))
print('MAE_rho:', np.mean(mae_rho))
print('MRE_vel:', np.mean(mre_v))
print('MRE_rho:', np.mean(mre_rho))
print('STD_vel:',np.std(mae_v))
print('STD_rho:',np.std(mae_rho))


# Skweness
print('skew_vel:', skew(mae_v))
print('skew_rho:',skew(mae_rho))
# kurtosis
print('kurtosis_vel:',kurtosis(mae_v))
print('kurtosis_rho:',kurtosis(mae_rho))



mpl.rcParams.update({'font.size': 19})
num_bins = 50  # Number of bins for the histogram

# Create a figure and two subplots
plt.figure(figsize=(14, 6))  # Adjusted width to 14 for better spacing

# First subplot for mae_v
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
plt.hist(mae_v, bins=num_bins, color='skyblue', edgecolor='black')
plt.xlabel('MAE - Velocity')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))


# Second subplot for mae_rho
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.hist(mae_rho, bins=num_bins, color='salmon', edgecolor='black')
plt.xlabel('MAE - Density')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))


# Display the plots
plt.tight_layout()  # Adjusts spacing to prevent overlap
plt.show()

index = 5

predicted_vx = predictions[0][index,:,:,0]
predicted_vy = predictions[1][index,:,:,0]
predicted_rho = predictions[2][index,:,:,0]

original_vx = Vx_test[index,:,:,0]
original_vy =Vy_test[index,:,:,0]
original_rho = Rho_test[index,:,:,0]

predicted_vx_av = predictions_av[0][index,:,:,0]
predicted_vy_av = predictions_av[1][index,:,:,0]
predicted_rho_av = predictions_av[2][index,:,:,0]

predicted_vx_w1 = predictions_w1[0][index,:,:,0]
predicted_vy_w1 = predictions_w1[1][index,:,:,0]
predicted_rho_w1 = predictions_w1[2][index,:,:,0]

predicted_vx_w2 = predictions_w2[0][index,:,:,0]
predicted_vy_w2 = predictions_w2[1][index,:,:,0]
predicted_rho_w2 = predictions_w2[2][index,:,:,0]


vel_mag = np.sqrt(predicted_vx**2+predicted_vy**2)
vel_mag_av = np.sqrt(predicted_vx_av**2+predicted_vy_av**2)
vel_mag_w1 = np.sqrt(predicted_vx_w1**2+predicted_vy_w1**2)
vel_mag_w2 = np.sqrt(predicted_vx_w2**2+predicted_vy_w2**2)

vel_mag_or = np.sqrt(original_vx**2+original_vy**2)


mpl.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(1, 5, figsize=(25, 5))

# Original Velocity Magnitude
im1 = axs[0].imshow(vel_mag_or, cmap='viridis', origin='lower')
axs[0].set_title('Original')
axs[0].axis('off')

# Simple
im2 = axs[1].imshow(vel_mag, cmap='viridis', origin='lower')
axs[1].set_title('Combination')
axs[1].axis('off')

# Average
im2 = axs[2].imshow(vel_mag_av, cmap='viridis', origin='lower')
axs[2].set_title('Average')
axs[2].axis('off')

# W1
im2 = axs[3].imshow(vel_mag_w1, cmap='viridis', origin='lower')
axs[3].set_title('Weighted 1')
axs[3].axis('off')

# W2
im2 = axs[4].imshow(vel_mag_w2, cmap='viridis', origin='lower')
axs[4].set_title('Weighted 2')
axs[4].axis('off')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Velocity Magnitude')

plt.tight_layout()
plt.show()

indexs= Input_test1[index].astype(int)
vel_mag_or = np.where(indexs, np.nan, vel_mag_or)
vel_mag = np.where(indexs, np.nan, vel_mag)
vel_mag_av = np.where(indexs, np.nan, vel_mag_av)
vel_mag_w1 = np.where(indexs, np.nan, vel_mag_w1)

predicted_vx = np.where(indexs, np.nan, predicted_vx)
predicted_vy = np.where(indexs, np.nan, predicted_vy)

original_vx = np.where(indexs, np.nan, original_vx)
original_vy = np.where(indexs, np.nan, original_vy)

predicted_vx_av = np.where(indexs, np.nan, predicted_vx_av)
predicted_vy_av = np.where(indexs, np.nan, predicted_vy_av)

predicted_vx_w1 = np.where(indexs, np.nan, predicted_vx_w1)
predicted_vy_w1 = np.where(indexs, np.nan, predicted_vy_w1)

predicted_vx_w2 = np.where(indexs, np.nan, predicted_vx_w2)
predicted_vy_w2 = np.where(indexs, np.nan, predicted_vy_w2)


margin_size = 10
expanded_mask = binary_dilation(indexs, iterations=margin_size)

# Set masked regions to NaN for streamline computation only
masked_predicted_vx = np.where(expanded_mask, np.nan, predicted_vx)
masked_predicted_vy = np.where(expanded_mask, np.nan, predicted_vy)

masked_original_vx = np.where(expanded_mask, np.nan, original_vx)
masked_original_vy = np.where(expanded_mask, np.nan, original_vy)

masked_predicted_vx_av = np.where(expanded_mask, np.nan, predicted_vx_av)
masked_predicted_vy_av = np.where(expanded_mask, np.nan, predicted_vy_av)

masked_predicted_vx_w1 = np.where(expanded_mask, np.nan, predicted_vx_w1)
masked_predicted_vy_w1 = np.where(expanded_mask, np.nan, predicted_vy_w1)

masked_predicted_vx_w2 = np.where(expanded_mask, np.nan, predicted_vx_w2)
masked_predicted_vy_w2 = np.where(expanded_mask, np.nan, predicted_vy_w2)

# Define the slice indices for zooming into a specific region
y_start, y_end = 0, 300
x_start, x_end = 0, 300


# Define grid for streamplot
Y, X = np.mgrid[y_start:y_end, x_start:x_end]

# Set font size for the plot
mpl.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(1, 5, figsize=(30, 6))

# Define a function to add a streamline plot to an axis
def add_streamline_plot(ax, vx, vy,vx_s,vy_s,title):
    vel_mag = np.sqrt(vx**2 + vy**2)
    im = ax.imshow(vel_mag, origin='lower', cmap='viridis', extent=[x_start, x_end, y_start, y_end])
    ax.streamplot(X - x_start, Y - y_start, vx_s, vy_s, color='white', density=1, arrowstyle='->', arrowsize=1.5, linewidth=1)
    ax.set_title(title)
    ax.axis('off')
    return im

# Original Velocity
im1 = add_streamline_plot(axs[0], original_vx[y_start:y_end, x_start:x_end], original_vy[y_start:y_end, x_start:x_end],
                          masked_original_vx[y_start:y_end, x_start:x_end], masked_original_vy[y_start:y_end, x_start:x_end], 'Original')

# Predicted Combination
add_streamline_plot(axs[1], predicted_vx[y_start:y_end, x_start:x_end], predicted_vy[y_start:y_end, x_start:x_end]
                    , masked_predicted_vx[y_start:y_end, x_start:x_end], masked_predicted_vy[y_start:y_end, x_start:x_end], 'Combination')

# Predicted Average
add_streamline_plot(axs[2], predicted_vx_av[y_start:y_end, x_start:x_end], predicted_vy_av[y_start:y_end, x_start:x_end]
                    , masked_predicted_vx_av[y_start:y_end, x_start:x_end], masked_predicted_vy_av[y_start:y_end, x_start:x_end], 'Average')

# Predicted Weighted 1
add_streamline_plot(axs[3], predicted_vx_w1[y_start:y_end, x_start:x_end], predicted_vy_w1[y_start:y_end, x_start:x_end]
                    , masked_predicted_vx_w1[y_start:y_end, x_start:x_end], masked_predicted_vy_w1[y_start:y_end, x_start:x_end], 'Weighted 1')

# Predicted Weighted 2
add_streamline_plot(axs[4], predicted_vx_w2[y_start:y_end, x_start:x_end], predicted_vy_w2[y_start:y_end, x_start:x_end]
                    , masked_predicted_vx_w2[y_start:y_end, x_start:x_end], masked_predicted_vy_w2[y_start:y_end, x_start:x_end], 'Weighted 2')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Velocity Magnitude')

plt.tight_layout()
plt.show()