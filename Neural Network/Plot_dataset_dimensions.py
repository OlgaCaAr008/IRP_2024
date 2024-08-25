
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator


project_folder = '/content/drive/My Drive/Colab Notebooks/dataset/final' # working folder path
os.chdir(project_folder) # changing the path

predictions25 = np.load('predictions_25.npy')
Vx_test25 = np.load('Vx_test_25.npy')
Vy_test25 = np.load('Vy_test_25.npy')
Rho_test25 = np.load('Rho_test_25.npy')

predictions50 = np.load('predictions_50.npy')
Vx_test50 = np.load('Vx_test_50.npy')
Vy_test50 = np.load('Vy_test_50.npy')
Rho_test50 = np.load('Rho_test_50.npy')

predictions75 = np.load('predictions_75.npy')
Vx_test75 = np.load('Vx_test_75.npy')
Vy_test75 = np.load('Vy_test_75.npy')
Rho_test75 = np.load('Rho_test_75.npy')

predictions100 = np.load('predictions_100.npy')
Vx_test100 = np.load('Vx_test_100.npy')
Vy_test100 = np.load('Vy_test_100.npy')
Rho_test100 = np.load('Rho_test_100.npy')

def compute_velocity_magnitude(vx, vy):
    return np.sqrt(vx**2 + vy**2)
def compute_mea(ground_truth,prediction):
  absolute_errors = np.abs(ground_truth - prediction)
  mae = np.mean(absolute_errors)
  return mae

predictions = predictions50 # predictions10 predictions100 predictions1000 predictions10000
Vx_test= Vx_test50
Vy_test= Vy_test50
Rho_test= Rho_test50

mae_v = []
mae_rho = []
mre_v = []
mre_rho = []

for j in range(predictions[0].shape[0]):
  predicted_vx = predictions[0][j]
  predicted_vy = predictions[1][j]
  predicted_velocity_magnitude = compute_velocity_magnitude(predicted_vx,predicted_vy)
  original_velocity_magnitude = compute_velocity_magnitude(Vx_test[j,:] , Vy_test[j,:] )
  predicted_rho = predictions[2][j]
  mae_v.append(compute_mea(original_velocity_magnitude,predicted_velocity_magnitude))
  mae_rho.append(compute_mea(Rho_test[j,:],predicted_rho))


  # Compute MRE for velocity and density, ignoring infinities
  mre_v_values = np.abs(original_velocity_magnitude - predicted_velocity_magnitude) / np.abs(original_velocity_magnitude)
  mre_rho_values = np.abs(Rho_test[j, :] - predicted_rho) / np.abs(Rho_test[j, :])
  mre_v_filtered = mre_v_values[np.isfinite(mre_v_values)]
  mre_rho_filtered = mre_rho_values[np.isfinite(mre_rho_values)]

  mre_v.append(np.mean(mre_v_filtered)*100)
  mre_rho.append(np.mean(mre_rho_filtered)*100)

print(np.mean(mre_v))
print(np.mean(mre_rho))


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

def post_process(predictions25,Vx_test25,Vy_test25,Rho_test25):
  rel_er_v = []


  for j in range(predictions25.shape[1]):
    vel = compute_velocity_magnitude(predictions25[0][j,:].reshape(300, 300), predictions25[1][j,:].reshape(300, 300))
    vel_or = compute_velocity_magnitude(Vx_test25[j,:].reshape(300, 300), Vy_test25[j,:].reshape(300, 300))
    vals = np.abs((vel- vel_or)/vel_or)
    filtered_vals = vals[np.isfinite(vals)]
    rel_er_v.append(np.mean(filtered_vals)*100)

  mean_v = np.mean(rel_er_v)

  return mean_v


mean_v25 = post_process(predictions25,Vx_test25,Vy_test25,Rho_test25)
mean_v50 = post_process(predictions50,Vx_test50,Vy_test50,Rho_test50)
mean_v75 = post_process(predictions75,Vx_test75,Vy_test75,Rho_test75)
mean_v100 = post_process(predictions100,Vx_test100,Vy_test100,Rho_test100)


print(mean_v25)
print(mean_v50)
print(mean_v100)
