
import os
import matplotlib.pyplot as plt
import numpy as np

########################### BLOCKAGE RATIO PLOTS #########################################################################
# Defining folder
project_folder = '/D_r' 
os.chdir(project_folder) 

# Importing the data
case0 = np.load('result_0.npy', allow_pickle=True)
case1 = np.load('result_1.npy', allow_pickle=True)
case2 = np.load('result_2.npy', allow_pickle=True)

# Arranging the data
time0, mean0_abs, max0, it0, vel0, r0  = np.array(case0[0]), case0[1], case0[2], np.array(case0[3]), case0[4], case0[5]
time1, mean2_abs, max1, it1, vel1, r1  = np.array(case1[0]), case1[1], case1[2], case1[3], case1[4], case1[5]
time2, mean1_abs, max2, it2, vel2, r2  = np.array(case2[0]), case2[1], case2[2], case2[3], case2[4], case2[5]

# ----------------------- Figures -------------------------------------------------
plt.figure(figsize=(10, 6), dpi=300)  
plt.rcParams.update({'font.size': 12})
time_intervals = np.linspace(time0.min(), time0.max(), num=5)  # Only 5 time intervals
iteration_ticks = np.interp(time_intervals, time0, it0)  # iteration ticks

# Plotting max values
plt.plot(time0/60, max0, 'o-', color='#377eb8', label=f'D = {r0*2}')  
plt.plot(time1/60, max1, '^-', color='#ff7f00', label=f'D = {r1*2}')  
plt.plot(time2/60, max2, 's-', color='#4daf4a', label=f'D = {r2*2}')  

plt.xlabel('Time (minutes)')
plt.ylabel('Maximum Absolute Change')
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.yscale('log')
plt.legend(title='Diameter', loc='upper right')

ax_secondary = plt.gca().twiny()  # secondary axis
ax_secondary.set_xticks(time_intervals / 60)  
ax_secondary.set_xticklabels([f'{int(it)}' for it in iteration_ticks])  # labels of the ticks
ax_secondary.set_xlabel('Iterations')
plt.tight_layout()
plt.show()


########################### DOMAIN DIMENSIONS PLOTS #########################################################################
# Defining folder
project_folder = '/DIMENSIONS' # working folder path
os.chdir(project_folder) # changing the path

# Importing the data
case0 = np.load('result_0.npy', allow_pickle=True)
case1 = np.load('result_1.npy', allow_pickle=True)
case2 = np.load('result_2.npy', allow_pickle=True)
case3 = np.load('result_3.npy', allow_pickle=True)

# Arranging the data
time0, mean0_abs, max0, it0, vel0, r0  = np.array(case0[0]), case0[1], case0[2], case0[3], case0[4], case0[5]
time1, mean1_abs, max1, it1, vel1, r1  = np.array(case1[0]), case1[1], case1[2], case1[3], case1[4], case1[5]
time2, mean2_abs, max2, it2, vel2, r2  = np.array(case2[0]), case2[1], case2[2], case2[3], case2[4], case2[5]
time3, mean3_abs, max3, it3, vel3, r3  = np.array(case3[0]), case3[1], case3[2], case3[3], case3[4], case3[5]

# ----------------------- Figures -------------------------------------------------
# PLOT vs TIME
plt.figure(figsize=(10, 6), dpi=300)  
plt.rcParams.update({'font.size': 12})

# Plot for Max Values
plt.plot(time0/60, max0, 'o-', color='#377eb8', label='6240x3120')  # Blue circles
plt.plot(time1/60, max1, '^-', color='#ff7f00', label='3120x1560')  # Orange triangles
plt.plot(time2/60, max2, 's-', color='#4daf4a', label='1560x780')  # Green squares
plt.plot(time3/60, max3, '>-', color='#e41a1c', label='780x390')  # Red triangles

plt.xlabel('Time (minutes)')
plt.ylabel('Maximum Absolute Change')
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.legend(title='Domain (Nx x Ny)', loc='upper right')
plt.tight_layout()
plt.show()

# PLOT vs ITERATIONS
plt.figure(figsize=(10, 6), dpi=300) 
plt.rcParams.update({'font.size': 12})

# Plot for Max Values
plt.plot(it0, max0, 'o-', color='#377eb8', label='6240x3120')  # Blue circles
plt.plot(it1, max1, '^-', color='#ff7f00', label='3120x1560')  # Orange triangles
plt.plot(it2, max2, 's-', color='#4daf4a', label='1560x780')  # Green squares
plt.plot(it3, max3, '>-', color='#e41a1c', label='780x390')  # Red triangles

plt.xlabel('Iterations')
plt.ylabel('Maximum Absolute Change')
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.yscale('log')
plt.xscale('log')
plt.legend(title='Domain (Nx x Ny)', loc='upper right')
plt.tight_layout()
plt.show()

# FLOW FIELD PLOT
plt.rcParams.update({'font.size': 20})

velocities = [vel0, vel1, vel2, vel3]

# Iterate over the velocity arrays to create each figure
for i, vel in enumerate(velocities):
    vel_mag = np.sqrt(vel[:,:,1]**2 + vel[:,:,2]**2)  # Calculate velocity magnitude

    # Create a figure
    plt.figure(figsize=(12, 5))
    img = plt.imshow(vel_mag, origin='lower', cmap='viridis')
    plt.colorbar(img, label='Velocity Magnitude') 
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

# STREAMLINES PLOT
plt.rcParams.update({'font.size': 20})

velocities = [vel1]  # Selecting one case

# Iterate over the velocity arrays to create each figure
for i, vel in enumerate(velocities):
    # Define the slice indices
    y_start, y_end = 700, 850
    x_start, x_end = 925, 1250

    # Extract the slice of the velocity field
    x_component = vel[y_start:y_end, x_start:x_end, 1]
    y_component = vel[y_start:y_end, x_start:x_end, 2]
    vel_mag = np.sqrt(x_component**2 + y_component**2)  

    # Assuming equal spacing 
    x_indices = np.linspace(x_start, x_end - 1, num=x_component.shape[1])
    y_indices = np.linspace(y_start, y_end - 1, num=y_component.shape[0])

    # Create figure 
    plt.figure(figsize=(18, 10))
    img = plt.imshow(vel_mag, origin='lower', cmap='viridis', extent=[x_indices.min(), x_indices.max(), y_indices.min(), y_indices.max()])
    plt.colorbar(img, label='Velocity Magnitude')  

    # Create streamlines 
    plt.streamplot(x_indices, y_indices, x_component, y_component, color='white', density=2, arrowstyle='->', arrowsize=1.5, linewidth=1)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

########################### 10 CASES PLOTS #########################################################################
# Defining folder
project_folder = '/final' # working folder path
os.chdir(project_folder) # changing the path

# Importing the data
case0 = np.load('result_0.npy', allow_pickle=True)
case1 = np.load('result_1.npy', allow_pickle=True)
case2 = np.load('result_2.npy', allow_pickle=True)
case3 = np.load('result_3.npy', allow_pickle=True)
case4 = np.load('result_4.npy', allow_pickle=True)
case5 = np.load('result_5.npy', allow_pickle=True)
case6 = np.load('result_6.npy', allow_pickle=True)
case7 = np.load('result_7.npy', allow_pickle=True)
case8 = np.load('result_8.npy', allow_pickle=True)
case9 = np.load('result_9.npy', allow_pickle=True)

# Arranging the data
time0, mean0_abs, max0, it0, vel0  = np.array(case0[0]), case0[1], case0[2], case0[3], case0[4]
time1, mean1_abs, max1, it1, vel1  = np.array(case1[0]), case1[1], case1[2], case1[3], case1[4]
time2, mean2_abs, max2, it2, vel2  = np.array(case2[0]), case2[1], case2[2], case2[3], case2[4]
time3, mean3_abs, max3, it3, vel3  = np.array(case3[0]), case3[1], case3[2], case3[3], case3[4]
time4, mean4_abs, max4, it4, vel4  = np.array(case4[0]), case4[1], case4[2], case4[3], case4[4]
time5, mean5_abs, max5, it5, vel5  = np.array(case5[0]), case5[1], case5[2], case5[3], case5[4]
time6, mean6_abs, max6, it6, vel6  = np.array(case6[0]), case6[1], case6[2], case6[3], case6[4]
time7, mean7_abs, max7, it7, vel7  = np.array(case7[0]), case7[1], case7[2], case7[3], case7[4]
time8, mean8_abs, max8, it8, vel8  = np.array(case8[0]), case8[1], case8[2], case8[3], case8[4]
time9, mean9_abs, max9, it9, vel9  = np.array(case9[0]), case9[1], case9[2], case9[3], case9[4]

# ----------------------- Figures -------------------------------------------------
# PLOT THE 10 CASES
plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), dpi=300)

# Define colors for each dataset
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00', '#000000']

# Plot for Max Values 
ax1 = axes[0]
datasets = [max0, max1, max2, max3, max4, max5, max6, max7, max8, max9]  # assuming max0 to max9 and time0 to time9 are defined
times = [time0, time1, time2, time3, time4, time5, time6, time7, time8, time9]

for i, (time, max_val) in enumerate(zip(times, datasets)):
    ax1.plot(time / 60, max_val, 'o-', color=colors[i], markersize=2)

ax1.set_xlabel('Time (minutes)')
ax1.set_ylabel('Maximum Absolute Change')
ax1.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax1.set_yscale('log')

ax2 = axes[1]
iterations = [it0, it1, it2, it3, it4, it5, it6, it7, it8, it9]

for i, (it, max_val) in enumerate(zip(iterations, datasets)):
    ax2.plot(it, max_val, 'o-', color=colors[i], markersize=2)

ax2.set_xlabel('Iterations')
ax2.set_ylabel('Maximum Absolute Change')
ax2.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()

# PLOT STREAMLINES
plt.rcParams.update({'font.size': 20})
velocities = [vel0, vel1, vel2, vel3, vel4, vel5, vel6, vel7, vel8, vel9]  # Assuming vel1 is defined and contains velocity data

# Iterate over the velocity arrays to create each figure
for i, vel in enumerate(velocities):
    # Define the slice indices
    y_start, y_end = 650, 850
    x_start, x_end = 850, 1200

    # Extract the slice of the velocity field
    x_component = vel[y_start:y_end, x_start:x_end, 1]
    y_component = vel[y_start:y_end, x_start:x_end, 2]
    vel_mag = np.sqrt(x_component**2 + y_component**2)  
    
    x_indices = np.linspace(x_start, x_end - 1, num=x_component.shape[1])
    y_indices = np.linspace(y_start, y_end - 1, num=y_component.shape[0])

    # Create figure 
    plt.figure(figsize=(18, 10))
    img = plt.imshow(vel_mag, origin='lower', cmap='viridis', extent=[x_indices.min(), x_indices.max(), y_indices.min(), y_indices.max()])
    plt.colorbar(img, label='Velocity Magnitude')  

    # Create streamlines
    plt.streamplot(x_indices, y_indices, x_component, y_component, color='white', density=2, arrowstyle='->', arrowsize=1.5, linewidth=1)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


# PLOT FIELD (VEL or RHO) DISTRIBUTIONS
vel_mag = np.sqrt(vel0[:,:,1]**2+vel0[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel0[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel1[:,:,1]**2+vel1[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel1[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude') 
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel2[:,:,1]**2+vel2[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel2[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel3[:,:,1]**2+vel3[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel3[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel4[:,:,1]**2+vel4[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel4[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel5[:,:,1]**2+vel5[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel5[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel6[:,:,1]**2+vel6[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel6[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel7[:,:,1]**2+vel7[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel7[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel8[:,:,1]**2+vel8[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel8[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

vel_mag = np.sqrt(vel9[:,:,1]**2+vel9[:,:,2]**2)
# Plotting
plt.figure(figsize=(12, 5))
plt.imshow(vel9[:,:,0], origin='lower', cmap='viridis')
plt.colorbar(label='Density Magnitude')  
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()
