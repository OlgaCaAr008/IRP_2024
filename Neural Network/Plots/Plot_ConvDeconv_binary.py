
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.stats import skew, kurtosis

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Conv-Deconv/simple' # working folder path
os.chdir(project_folder) # changing the path

predictions1 = np.load('predictions1.npy')
Vx_test1 = np.load('Vx_test1.npy')
Vy_test1 = np.load('Vy_test1.npy')
Rho_test1 = np.load('Rho_test1.npy')
test_scores = np.load('test_scores1.npy')
print('---------------- Simple -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])

history_data = np.load('training_history_data1.npz')

epochs1 = history_data['epochs']
loss1 = history_data['loss']
val_loss1 = history_data['val_loss']

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Conv-Deconv/output' # working folder path
os.chdir(project_folder) # changing the path
predictions2 = np.load('predictions2.npy')
test_scores = np.load('test_scores2.npy')
print('---------------- Dense layer at the output -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])

history_data = np.load('training_history_data2.npz')

epochs2 = history_data['epochs']
loss2 = history_data['loss']
val_loss2 = history_data['val_loss']

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Conv-Deconv/middle' # working folder path
os.chdir(project_folder) # changing the path
predictions3 = np.load('predictions1.npy')
test_scores = np.load('test_scores1.npy')
print('---------------- Dense layer in the middle -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])

history_data = np.load('training_history_data1.npz')

epochs3 = history_data['epochs']
loss3 = history_data['loss']
val_loss3 = history_data['val_loss']

project_folder = '/content/drive/My Drive/Colab Notebooks/ARCHER2_RUNS/results/Conv-Deconv/both' # working folder path
os.chdir(project_folder) # changing the path
predictions4 = np.load('predictions2.npy')
test_scores = np.load('test_scores2.npy')
print('---------------- Dense layer middle and output -------------')
print('Total loss:',test_scores[0])
print('Vx loss:',test_scores[1])
print('Vy loss:',test_scores[2])
print('Rho loss:',test_scores[3])
print('Vx mea:',test_scores[4])
print('Vy mea:',test_scores[5])
print('Rho mea:',test_scores[6])

history_data = np.load('training_history_data2.npz')

epochs4 = history_data['epochs']
loss4 = history_data['loss']
val_loss4 = history_data['val_loss']

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)
def compute_mea(ground_truth,prediction):
  absolute_errors = np.abs(ground_truth - prediction)
  mae = np.mean(absolute_errors)
  return mae

predictions = predictions4 # predictions10 predictions100 predictions1000 predictions10000
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

# Second subplot for mae_rho
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
plt.hist(mae_rho, bins=num_bins, color='salmon', edgecolor='black')
plt.xlabel('MAE - Density')
plt.ylabel('Frequency')
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))


# Display the plots
plt.tight_layout()  # Adjusts spacing to prevent overlap
plt.show()


# Increase default font size for all plot elements
mpl.rcParams.update({'font.size': 17})

plt.figure(figsize=(12, 6))
plt.plot(epochs1, loss1, color='blue', linestyle='-', label='Training Loss - Simple case')  # Dark red solid line for training loss
plt.plot(epochs1, val_loss1, color='deepskyblue', linestyle='--', label='Validation Loss - Simple case')  # Salmon dashed line for validation loss

plt.plot(epochs2, loss2, color='green', linestyle='-', label='Training Loss - Output case')  # Dark red solid line for training loss
plt.plot(epochs2, val_loss2, color='mediumseagreen', linestyle='--', label='Validation Loss  - Output case')  # Salmon dashed line for validation loss

plt.plot(epochs3, loss3, color='darkred', linestyle='-', label='Training Loss - Middle case')  # Dark red solid line for training loss
plt.plot(epochs3, val_loss3, color='salmon', linestyle='--', label='Validation Loss - Middle case')  # Salmon dashed line for validation loss

plt.plot(epochs4, loss4, color='purple', linestyle='-', label='Training Loss - Combined case')  # Dark red solid line for training loss
plt.plot(epochs4, val_loss4, color='orchid', linestyle='--', label='Validation Loss - Combined case')  # Salmon dashed line for validation loss
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.yscale('log')  # Logarithmic scale for loss
plt.show()  # Display the figure

index = 1

# Simple
predicted_vx10 = predictions1[0][index,:,:,0]
predicted_vy10 = predictions1[1][index,:,:,0]
predicted_rho10 = predictions1[2][index,:,:,0]

# Dense layer at the output
predicted_vx100 = predictions2[0][index,:].reshape(300, 300)
predicted_vy100 = predictions2[1][index,:].reshape(300, 300)
predicted_rho100 = predictions2[2][index,:].reshape(300, 300)

# Dense layer in the middle
predicted_vx1000 = predictions3[0][index,:,:,0]
predicted_vy1000 = predictions3[1][index,:,:,0]
predicted_rho1000 = predictions3[2][index,:,:,0]

# Dense layer middle and output
predicted_vx10000 = predictions4[0][index,:].reshape(300, 300)
predicted_vy10000 = predictions4[1][index,:].reshape(300, 300)
predicted_rho10000 = predictions4[2][index,:].reshape(300, 300)

original_vx = Vx_test1[index,:,:,0]
original_vy =Vy_test1[index,:,:,0]
original_rho = Rho_test1[index,:,:,0]

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)

original_velocity_magnitude = compute_velocity_magnitude(original_vx, original_vy)
predicted_velocity_magnitude10 = compute_velocity_magnitude(predicted_vx10, predicted_vy10) # Simple
predicted_velocity_magnitude100 = compute_velocity_magnitude(predicted_vx100, predicted_vy100) # Dense layer at the output
predicted_velocity_magnitude1000 = compute_velocity_magnitude(predicted_vx1000, predicted_vy1000) # Dense layer in the middle
predicted_velocity_magnitude10000 = compute_velocity_magnitude(predicted_vx10000, predicted_vy10000) # Dense layer middle and output


mpl.rcParams.update({'font.size': 20})


fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns

# Original Velocity Magnitude
im1 = axs[0].imshow(original_velocity_magnitude, cmap='viridis',origin='lower')
axs[0].set_title('Original')
axs[0].axis('off')

# Simple
im2 = axs[1].imshow(predicted_velocity_magnitude10, cmap='viridis',origin='lower')
axs[1].set_title('No Dense layer')
axs[1].axis('off')

im3 = axs[2].imshow(predicted_velocity_magnitude100, cmap='viridis',origin='lower')
axs[2].set_title('Before output')
axs[2].axis('off')

im4 = axs[3].imshow(predicted_velocity_magnitude1000, cmap='viridis',origin='lower')
axs[3].set_title('Between Conv-Deconv')
axs[3].axis('off')

im5 = axs[4].imshow(predicted_velocity_magnitude10000, cmap='viridis',origin='lower')
axs[4].set_title('Between and before output')
axs[4].axis('off')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Velocity Magnitude')

plt.tight_layout()
plt.show()


# Increase default font size for all plot elements
mpl.rcParams.update({'font.size': 20})

# Setup figure and subplots for 5 images
fig, axs = plt.subplots(1, 5, figsize=(25, 5))  # 1 row, 5 columns

# Original Velocity Magnitude
im1 = axs[0].imshow(original_rho, cmap='viridis',origin='lower')
axs[0].set_title('Original')
axs[0].axis('off')  # Hide axes for better visualization

# Predicted Velocity Magnitudes at different iterations
im2 = axs[1].imshow(predicted_rho10, cmap='viridis',origin='lower')
axs[1].set_title('No Dense layer')
axs[1].axis('off')

im3 = axs[2].imshow(predicted_rho100, cmap='viridis',origin='lower')
axs[2].set_title('Before output')
axs[2].axis('off')

im4 = axs[3].imshow(predicted_rho1000, cmap='viridis',origin='lower')
axs[3].set_title('Between Conv-Deconv')
axs[3].axis('off')

im5 = axs[4].imshow(predicted_rho10000, cmap='viridis',origin='lower')
axs[4].set_title('Between and before output')
axs[4].axis('off')

# Place a single colorbar at the bottom of the plots
cbar_ax = fig.add_axes([0.15, -0.05, 0.7, 0.05])  # x-position, y-position, width, height
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.012, pad=0.04, label='Density')

plt.tight_layout()
plt.show()