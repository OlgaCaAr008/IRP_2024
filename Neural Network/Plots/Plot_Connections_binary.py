
import os
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


drive.mount('/content/drive')

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Connecting/Binary/con_input_out_maxpool' # working folder path
os.chdir(project_folder) # changing the path

predictions1 = np.load('predictions2.npy')
Vx_test1 = np.load('Vx_test2.npy')
Vy_test1 = np.load('Vy_test2.npy')
Rho_test1 = np.load('Rho_test2.npy')
test_scores = np.load('test_scores2.npy')
print('---------------- Connecting input-output with Maxpooling -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])


project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Connecting/Binary/U-Net' # working folder path
os.chdir(project_folder) # changing the path
predictions2 = np.load('predictions3.npy')
test_scores = np.load('test_scores3.npy')
print('---------------- U-Net -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)
def compute_mea(ground_truth,prediction):
  absolute_errors = np.abs(ground_truth - prediction)
  mae = np.mean(absolute_errors)
  return mae

predictions = predictions2 # predictions10 predictions100 predictions1000 predictions10000
Vx_test = Vx_test1
Vy_test = Vy_test1
Rho_test = Rho_test1

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

index = 1

# Simple
predicted_vx10 = predictions1[0][index,:].reshape(300, 300)
predicted_vy10 = predictions1[1][index,:].reshape(300, 300)
predicted_rho10 = predictions1[2][index,:].reshape(300, 300)

# Dense layer at the output
predicted_vx100 = predictions2[0][index,:].reshape(300, 300)
predicted_vy100 = predictions2[1][index,:].reshape(300, 300)
predicted_rho100 = predictions2[2][index,:].reshape(300, 300)

original_vx = Vx_test1[index,:].reshape(300, 300)
original_vy =Vy_test1[index,:].reshape(300, 300)
original_rho = Rho_test1[index,:].reshape(300, 300)

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)

original_velocity_magnitude = compute_velocity_magnitude(original_vx, original_vy)
predicted_velocity_magnitude10 = compute_velocity_magnitude(predicted_vx10, predicted_vy10) # Simple
predicted_velocity_magnitude100 = compute_velocity_magnitude(predicted_vx100, predicted_vy100) # Dense layer at the output

mpl.rcParams.update({'font.size': 20})


fig, axs = plt.subplots(1, 3, figsize=(25, 8))  # 1 row, 5 columns

# Original Velocity Magnitude
im1 = axs[0].imshow(original_velocity_magnitude, cmap='viridis',origin='lower')
axs[0].set_title('Original')
axs[0].axis('off')

# Simple
im2 = axs[1].imshow(predicted_velocity_magnitude10, cmap='viridis',origin='lower')
axs[1].set_title('Connecting I/O')
axs[1].axis('off')

im3 = axs[2].imshow(predicted_velocity_magnitude100, cmap='viridis',origin='lower')
axs[2].set_title('U-Net')
axs[2].axis('off')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Velocity Magnitude')

plt.tight_layout()
plt.show()

# Increase default font size for all plot elements
mpl.rcParams.update({'font.size': 20})

# Setup figure and subplots for 5 images
fig, axs = plt.subplots(1, 3, figsize=(25, 8))  # 1 row, 5 columns

# Original Velocity Magnitude
im1 = axs[0].imshow(original_rho, cmap='viridis',origin='lower')
axs[0].set_title('Original')
axs[0].axis('off')  # Hide axes for better visualization

# Predicted Velocity Magnitudes at different iterations
im2 = axs[1].imshow(predicted_rho10, cmap='viridis',origin='lower')
axs[1].set_title('Connecting I/O')
axs[1].axis('off')

im3 = axs[2].imshow(predicted_rho100, cmap='viridis',origin='lower')
axs[2].set_title('U-Net')
axs[2].axis('off')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Density')

plt.tight_layout()
plt.show()