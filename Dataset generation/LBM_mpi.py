
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
def LBM_solver(Nx,Ny,idxs,cxs,cys,weights,F,geo,tau):
  store_variables = np.zeros((Ny,Nx,3))

  # Streaming
  for i, cx, cy in zip(idxs, cxs, cys):
    F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
    F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)

  # Calculate fluid variables (no need of computing pressure)
  rho = np.sum(F,2)
  P = 1/3 * rho
  ux  = np.sum(F*cxs,2) / rho
  uy  = np.sum(F*cys,2) / rho

  ux[geo] = 0
  uy[geo] = 0

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
  Nx, Ny, Nt, Nx_sd, Ny_sd, uMax, NL, cxs, cys, idxs, rho0, weights, Re, stop_criteria, obst_maxX, obst_maxY, cnt_x, cnt_y  = args

  geometry, ymin_geom, ymax_geom, xmin_geom, xmax_geom = geometry_creation(Nx, Ny, obst_maxX, obst_maxY, cnt_x, cnt_y)

  lim_x_low, lim_x_high  = int(cnt_x - Nx_sd //2.5), int(cnt_x + (Nx_sd -Nx_sd //2.5))
  lim_y_low, lim_y_high  = cnt_y - Ny_sd //2, cnt_y + (Ny_sd -Ny_sd //2)

  # Initilise the simulation
  F = initialise(Nx,Ny,uMax,NL,cxs,cys,rho0,weights)

  nu = uMax*2*(xmax_geom-xmin_geom)/Re # kinematic viscosity
  tau = 3*nu+0.5 # collision timescale

  # Initilise arrays
  ux_history = np.zeros((Ny_sd, Nx_sd, 2))
  uy_history = np.zeros((Ny_sd, Nx_sd, 2))

  change_hist = []

  # Simulation Main Loop
  for it in range(Nt):
    F, store_variables = LBM_solver(Nx,Ny,idxs,cxs,cys,weights,F,geometry,tau)

    # Monitoring velocity changes
    ux_history[:, :, 0] = ux_history[:, :, 1]
    uy_history[:, :, 0] = uy_history[:, :, 1]
    ux_history[:, :, 1] = store_variables[lim_y_low : lim_y_high ,lim_x_low : lim_x_high ,1]
    uy_history[:, :, 1] = store_variables[lim_y_low : lim_y_high ,lim_x_low : lim_x_high, 2]

    if it > 0:
      current = np.sqrt(ux_history[:, :, 1]**2 +uy_history[:, :, 1]**2)
      previous = np.sqrt(ux_history[:, :, 0]**2 + uy_history[:, :, 0]**2)
      error_max = np.nanmax(current - previous)
      change_hist.append(error_max)

      if len(change_hist)>3:
        if change_hist[-1] < stop_criteria and change_hist[-2] < stop_criteria and change_hist[-3] < stop_criteria and change_hist[-1]<change_hist[-2] and change_hist[-2]<change_hist[-3]:
          break

  return store_variables[lim_y_low : lim_y_high ,lim_x_low : lim_x_high , :], it, geometry[lim_y_low : lim_y_high ,lim_x_low : lim_x_high]


from mpi4py import MPI
import numpy as np
import time

start_time = time.time()
# Define the inputs
Nx, Ny, Nt = 3000, 1500, 2500
uMax, Re, rho0 = 0.1, 20, 1.0
obst_maxX, obst_maxY = 150, 150  # Obstacle box limits
cnt_x, cnt_y = Nx // 3, Ny // 2  # Center of the object
stop_criteria = 8e-6

Nx_sd, Ny_sd = 500, 450

# Lattice speeds / weights
NL = 9
idxs = np.arange(NL)
cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

DATASET_DIM = 16

# Initialize the MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank() # identifier for each process to assign tasks
size = comm.Get_size() # processes (nº of parallel runs)

if rank == 0: # master
    indices = np.array_split(np.arange(DATASET_DIM), size)
else:
    indices = None

local_indices = comm.scatter(indices, root=0)

# Assign the parameters and compute
local_params_list = [(Nx, Ny, Nt, Nx_sd, Ny_sd, uMax, NL, cxs, cys, idxs, rho0, weights, Re, stop_criteria,obst_maxX, obst_maxY, cnt_x, cnt_y) for j in range(len(local_indices))]
local_results = [MainComputation(params) for params in local_params_list]

store_geom = []
store_field = []
for i in range(len(local_results)):
  if local_results[i][1]<Nt-1:
    store_geom.append(local_results[i][2])
    store_field.append(local_results[i][0])

# Gather geometries and fields from all processes at the root
gathered_geoms = comm.gather(store_geom, root=0)
gathered_fields = comm.gather(store_field, root=0)

if rank == 0:
    # Flatten the lists of geometries and fields
    all_geometries = [geom for sublist in gathered_geoms for geom in sublist]
    all_fields = [field for sublist in gathered_fields for field in sublist]

    # Save the data to files using NumPy
    np.save('all_geometries.npy', np.array(all_geometries))
    np.save('all_fields.npy', np.array(all_fields))
    end_time = time.time()
    np.save('time.npy', end_time-start_time)
