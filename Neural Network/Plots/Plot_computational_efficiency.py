
import matplotlib.pyplot as plt
import numpy as np

# Data
mean_relative_error_rho = np.array([0.20, 0.27, 0.74, 0.88, 0.27,0.27,0.29,
                                    0.27,0.53,0.23,0.27,0.19,0.21,0.20,0.21,0.17])
mean_relative_error_vel = np.array([11.62, 26.48, 212.16, 354.37, 28.10,26.87, 54.14,27.51,
                                    32.01,18.44,26.28,15.34,16.18,15.83,17.66,11.44])
training_time = np.array([46, 596, 580, 480, 343, 841, 882, 511, 60, 151,
                          150,118,159,96,83,271])
labels = ['4-layer',
          'Dense output - Binary',
          'Simple - Binary',
          'Dense Middle - Binary',
          'Dense Middle/output Dense - Binary',
          'Dense output - SDF',
          'Simple - SDF',
          'Dense Middle/output -SDF',
          'Connecting I/O - Binary',
          'U-Net + Dense - Binary',
          'U-Net + Dense - SDF',
          'U-Net - SDF',
          'Concatenated',
          'Average','Weighted-1','Weighted-2']

# Define colors and markers
colors = plt.cm.viridis(np.linspace(0, 0.85, len(labels)))  # Exclude the yellow end (0.85 and above)
markers = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X', 'H', '*', 'd', 'p', '|', '_', '+']

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First plot for mean_relative_error_vel
for i in range(len(labels)):
    axs[0].scatter(training_time[i], mean_relative_error_vel[i], color=colors[i], marker=markers[i], s=100, alpha=0.7, label=labels[i])

axs[0].set_xlabel('Training Time (min)', fontsize=18)
axs[0].set_ylabel('MRE (%) Vel', fontsize=18)
axs[0].set_yscale('log')
axs[0].grid(True)

# Second plot for mean_relative_error_rho
for i in range(len(labels)):
    axs[1].scatter(training_time[i], mean_relative_error_rho[i], color=colors[i], marker=markers[i], s=100, alpha=0.7, label=labels[i])

axs[1].set_xlabel('Training Time (min)', fontsize=18)
axs[1].set_ylabel('MRE (%) Rho', fontsize=18)
axs[1].grid(True)

# Create a single legend for both plots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=16)

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.3)

# Show the plot
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data
mean_relative_error_rho = np.array([0.17,0.2,0.22,0.21])
mean_relative_error_vel = np.array([11.44,13.12,13.60,14.74])
training_time = np.array([271,329,99,70])
labels = ['100%', '75%','50%','25%']

# Define colors and markers
colors = plt.cm.viridis(np.linspace(0, 0.85, len(labels)))  # Exclude the yellow end (0.85 and above)
markers = ['o', 's', 'D', '^']

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# First plot for mean_relative_error_vel
for i in range(len(labels)):
    axs[0].scatter(training_time[i], mean_relative_error_vel[i], color=colors[i], marker=markers[i], s=100, alpha=0.7, label=labels[i])

axs[0].set_xlabel('Training Time (min)', fontsize=18)
axs[0].set_ylabel('MRE (%) Vel', fontsize=14)
axs[0].grid(True)

# Second plot for mean_relative_error_rho
for i in range(len(labels)):
    axs[1].scatter(training_time[i], mean_relative_error_rho[i], color=colors[i], marker=markers[i], s=100, alpha=0.7, label=labels[i])

axs[1].set_xlabel('Training Time (min)', fontsize=18)
axs[1].set_ylabel('MRE (%) Rho', fontsize=18)
axs[1].grid(True)

# Create a single legend for both plots
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=16)

# Adjust layout
plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.15, wspace=0.3)

# Show the plot
plt.show()
