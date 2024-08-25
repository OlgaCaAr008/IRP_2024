
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import csv  #
import os
import math
from shapely.geometry import Point, Polygon
from scipy.ndimage import gaussian_filter


# Change to working directory
project_folder = '/2D' # example
os.chdir(project_folder) 

# INITIAL STARTING GEOMETRY ----------------------------------------------------
profile_geom= [[45.0, 15.0],
    [42.25, 25.56],
    [35.0, 32.81],
    [25.0, 35.0],
    [15.0, 32.81],
    [7.75, 25.56],
    [5.0, 15.0],
    [7.75, 4.44],
    [15.0, -2.81],
    [25.0, -5.0],
    [35.0, -2.81],
    [42.25, 4.44],
    [45.0, 15.0]  ]# [x-coordinate z-coordinate] in meters

# INPUT DATA -------------------------------------------------------------------
resolution= 1 # nº pixels/meter
polygon = Polygon(profile_geom)
margin= 15 # meters of margin around the geometry

# PIXELS matrix ----------------------------------------------------------------
# Range
max_values = np.max(profile_geom, axis=0)
min_values = np.min(profile_geom, axis=0)

# Number of pixels
x_pixels=math.ceil(resolution*(margin*2+max_values[0]-min_values[0])) # width + margin
z_pixels=math.ceil(resolution*(margin*2+max_values[1]-min_values[1])) # height + margin

# Creating the matrix with 0
image=np.zeros((z_pixels,x_pixels))

for i in range(x_pixels):
    for j in range(z_pixels):
        # Convert current pixel to coordinates
        x_pixel = (min_values[0]-margin) + i / resolution
        z_pixel = (min_values[1]-margin) + j / resolution

        # Check if the point is inside the polygon
        if polygon.contains(Point(x_pixel, z_pixel)):
            image[j,i] = 1  #  1 if inside the geometry

plt.figure(figsize=(10, 6))
plt.imshow(image, cmap='gray', origin='lower')
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')
plt.show()

# Use skimage to find contours at a half-level (0.5 is an example threshold)
contours = measure.find_contours(image, 0.5)

contours_array = contours[0]
x_vals = contours_array[:,1]
y_vals = contours_array[:,0]

sigma = 2  # amount of smoothing
x_vals_smooth = gaussian_filter(x_vals, sigma=5)
y_vals_smooth = gaussian_filter(y_vals, sigma=5)

x_vals_smooth = np.append(x_vals_smooth, x_vals_smooth[0])
y_vals_smooth = np.append(y_vals_smooth, y_vals_smooth[0])


plt.figure(figsize=(10, 8))  # Set the figure size
plt.imshow(image, cmap='gray', interpolation='none', origin='lower')  # Display the image

#plt.plot(x_vals, y_vals, 'r-', linewidth=2)
plt.plot(x_vals_smooth, y_vals_smooth, 'b-', linewidth=2, label='Smoothed Contour')

# Enhancing the plot
plt.xlabel('X Pixels')
plt.ylabel('Y Pixels')
plt.show()

z_coordinates = [0] * len(x_vals_smooth)
x_coordinates = [x / resolution for x in x_vals_smooth]
y_coordinates = [y / resolution for y in y_vals_smooth]
rows = zip(x_coordinates, y_coordinates, z_coordinates)

# Specify the CSV file path
filename = 'smoothed_coordinates.csv'

# Write data to CSV
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for row in rows:
        writer.writerow(row)
