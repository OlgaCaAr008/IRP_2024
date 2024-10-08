
########################### FUNCTIONS #########################################################################
# Ordering the points
def order_points(points):
    # Calculate centroid of the points
    centroid = np.mean(points, axis=0)
    # Compute angles from centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    # Sort points by angles
    return points[np.argsort(angles)]

# Is point inside the polygon: https://people.utm.my/shahabuddin/?p=6277
def is_point_in_polygon(point, vertices):
    num_intersections = 0
    num_vertices = len(vertices)

    for i in range(num_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % num_vertices]  # wrap around to first vertex for last edge
        if ((v1[1] > point[1]) != (v2[1] > point[1])) and \
                (point[0] < (v2[0] - v1[0]) * (point[1] - v1[1]) / (v2[1] - v1[1]) + v1[0]):
            num_intersections += 1

    return num_intersections % 2 == 1

# Creating the geometry
def geometry_creation(Nx,Ny,obst_maxX,obst_maxY,cnt_x,cnt_y):

  geo = np.zeros((Nx, Ny), dtype=bool)

  min_area = obst_maxX * obst_maxY*0.1 # minimum area is a 10% of the max area
  polygon_area = 0

  # Generating the geometry and ensuring area constrain
  while polygon_area < min_area:
    numPoints = np.random.randint(3, 10)
    points = np.column_stack((np.random.randint(cnt_x-round(obst_maxX/2), cnt_x+round(obst_maxX/2), numPoints),
                            np.random.randint(cnt_y-round(obst_maxY/2), cnt_y+round(obst_maxY/2), numPoints)))
    ordered_points = order_points(points)
    polygon = ordered_points.tolist()

  # Filling the geometry array
    for x in range(Nx):
      for y in range(Ny):
        if is_point_in_polygon((x,y),polygon):
          geo[x,y] = True

    object_indices = np.where(geo)
    polygon_area = len(object_indices[0])

  geo = geo.T

  object_indices = np.where(geo)

  # Minimum and maximum coordinates of the geometry
  y_indices, x_indices = object_indices
  ymin_f = int(np.min(object_indices[0]))
  ymax_f = int(np.max(object_indices[0]))
  xmin_f = int(np.min(object_indices[1]))
  xmax_f = int(np.max(object_indices[1]))

  return geo, ymin_f, ymax_f, xmin_f, xmax_f

# Initialise the LBM
def initialise(Nx,Ny,uMax,NL,cxs,cys,rho0,weights):
  # Initial Conditions - uniform flow at input
  ux_initial = np.full((Ny, Nx), uMax)  # Uniform horizontal velocity
  uy_initial = np.zeros((Ny, Nx))  # No vertical velocity

  # Initial F
  F = np.zeros((Ny,Nx,NL))
  for i in range(NL):
    cu = 3 * (cxs[i] * ux_initial + cys[i] * uy_initial)
    F[:, :, i] = rho0 * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux_initial**2 + uy_initial**2))

  return F

# LBM solver
def LBM_solver(Nx,Ny,idxs,cxs,cys,weights,F,cylinder,tau):
  store_variables = np.zeros((Ny,Nx,3))

  # Streaming
  for i, cx, cy in zip(idxs, cxs, cys):
    F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
    F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

  # Calculate fluid variables
  rho = np.sum(F,2)
  P = 1/3 * rho
  ux  = np.sum(F*cxs,2) / rho
  uy  = np.sum(F*cys,2) / rho

  # Store variables
  ux[cylinder] = 0
  uy[cylinder] = 0

  # Store variables
  store_variables[:,:,0] = rho
  store_variables[:,:,1] = ux
  store_variables[:,:,2] = uy

  # Apply Collision
  Feq = np.zeros_like(F)
  for i, cx, cy, w in zip(idxs, cxs, cys, weights):
    cu = 3 * (cxs[i] * ux + cys[i] * uy)
    Feq[:,:,i]  = rho * weights[i] * (1 + cu + 0.5 * cu**2 - 1.5 * (ux**2 + uy**2))

  F += -(1.0/tau) * (F - Feq)

  # Apply periodic boundary conditions
  F[:, -1, :] = F[:, 0, :]
  F[:, 0, :] = F[:, -1, :]
  F[-1, :, :] = F[0, :, :]
  F[0, :, :] = F[-1, :, :]

  return F, store_variables

# Main computation
def MainComputation(args):
  Nx, Ny, Nt, uMax, NL, cxs, cys, idxs, rho0, weights, Re, geometry, xmax_geom, xmin_geom, ymax_subd, ymin_subd, xmax_subd, xmin_subd, it_saved, freq = args

  # Initilise the simulation
  F = initialise(Nx,Ny,uMax,NL,cxs,cys,rho0,weights)

  nu = uMax*2*(xmax_geom-xmin_geom)/Re # kinematic viscosity
  tau = 3*nu+0.5 # collision timescale

  # Initilise arrays
  save = np.zeros((ymax_subd-ymin_subd,xmax_subd-xmin_subd,it_saved))
  ux_history = np.zeros((ymax_subd-ymin_subd, xmax_subd-xmin_subd, 2))
  uy_history = np.zeros((ymax_subd-ymin_subd, xmax_subd-xmin_subd, 2))
  mean_vel_changes = []
  max_vel_changes = []

  report_time = []
  report_mean_abs = []
  report_max = []
  report_it = []

  start_time = time.time()

  # Simulation Main Loop
  for it in range(Nt):
    print(it)
    F, store_variables = LBM_solver(Nx,Ny,idxs,cxs,cys,weights,F,geometry,tau)

    # Monitoring velocity changes
    ux_history[:, :, 0] = ux_history[:, :, 1]
    uy_history[:, :, 0] = uy_history[:, :, 1]
    ux_history[:, :, 1] = store_variables[ymin_subd:ymax_subd,xmin_subd:xmax_subd,1]
    uy_history[:, :, 1] = store_variables[ymin_subd:ymax_subd,xmin_subd:xmax_subd,2]

    if it > 0:
      current = np.sqrt(ux_history[:, :, 1]**2 +uy_history[:, :, 1]**2)
      previous = np.sqrt(ux_history[:, :, 0]**2 + uy_history[:, :, 0]**2)
      error_vel = np.nanmean(np.abs((current - previous) / previous)) * 100
      error_max = np.nanmax(current - previous)
      mean_vel_changes.append(error_vel)
      max_vel_changes.append(error_max)

      current_time = time.time()

      if it % freq == 0 and it/freq<it_saved: # just save the velocity magnitude
        report_time.append(current_time - start_time)
        report_mean_abs.append(mean_vel_changes[-1])
        report_max.append(max_vel_changes[-1])
        report_it.append(it)

  return report_time, report_mean_abs, report_max, report_it, store_variables


import numpy as np
import time
from multiprocessing import Pool, cpu_count

# Input data
Nx, Ny, Nt = 3000, 1500, 2500
cnt_x, cnt_y = Nx // 3, Ny // 2  # Center of the object
obst_maxX, obst_maxY = 150, 150 # Obstacle box limits
xmax_subd = int(cnt_x+obst_maxX/2+3*obst_maxX)
xmin_subd = int(cnt_x-obst_maxX/2-2*obst_maxX)
ymax_subd = int(cnt_y+obst_maxY/2+2*obst_maxX)
ymin_subd =  int(cnt_y-obst_maxY/2-2*obst_maxX)

uMax, Re, rho0 = 0.1, 20, 1.0

dx, dy, dt = 1, 1, 1

it_saved = 200
freq = Nt // it_saved

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0,-1,-1,-1])
cys = np.array([0, 1, 1, 0,-1,-1,-1, 0, 1])
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36]) # sums to 1

DATASET_DIM = 10 # example

geometries_array = []
xmax = []
xmin = []
ymin = []
ymax = []

start_time = time.time()

# Creating all the geometries
for i in range(DATASET_DIM):
  geo, ymin_f, ymax_f, xmin_f, xmax_f = geometry_creation(Nx,Ny,obst_maxX,obst_maxY,cnt_x,cnt_y)
  geometries_array.append(geo)
  xmax.append(xmax_f)
  xmin.append(xmin_f)
  ymin.append(ymin_f)
  ymax.append(ymax_f)

# Defining all the parameters for the simulations
params_list = [(Nx, Ny, Nt, uMax, NL, cxs, cys, idxs, rho0, weights, Re, geometries_array[j], xmax[j], xmin[j], ymax_subd, ymin_subd, xmax_subd, xmin_subd, it_saved, freq) for j in range(DATASET_DIM)]

with Pool(len(params_list)) as pool:
    results = pool.map(MainComputation, params_list)

for index, result in enumerate(results):
    filename = f"result_{index}.npy"
    np.save(filename, result)
