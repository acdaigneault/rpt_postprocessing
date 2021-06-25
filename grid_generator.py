import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reactor dimensions
L = 0.3 # m
R = 0.1 # m

# Detector dimensions
r = 0.0381 # m
l = 0.0762 # m

# Position of the detector
FP = [0.2, 0, 0.075]
MP = [FP[0] + l/2, FP[1], FP[2]]
print(FP, MP)

# Positions specifications
n_point = 300 # Total number of positions
n_point_per_stage = 25
n_stage = int(n_point/n_point_per_stage) # Number of stages to get n_point
radial_distance = 0.03 # Radial distance between positions
buffer_top_bottom = 0.02 # Space between the bottom or top to the close stage
stage_height = (L - 2*buffer_top_bottom)/(n_stage - 1)
angles = np.arange(0, 2*np.pi, np.pi/4)

# Positions generation
positions = np.empty((0,3), int)
for i in range(0, n_stage):
    positions = np.append(positions, np.array([[0, 0 ,i * stage_height + buffer_top_bottom]]), axis=0)
    for j in angles:
        for k in [1, 2, 3]:
            x = k * radial_distance * np.sin(j)
            y = k * radial_distance * np.cos(j)
            positions = np.append(positions, np.array([[x, y, i* stage_height + buffer_top_bottom]]), axis=0)


# Function to generate reactor, detector
def data_for_cylinder(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

# Mesh generation
Xr, Yr, Zr = data_for_cylinder(0,0,R,L)
Zd, Yd, Xd = data_for_cylinder(FP[2],FP[1],r,l)

# Plots
ax = plt.axes(projection='3d')

ax.plot_surface(Xr, Yr, Zr, alpha=0.2, color="grey", label="Reactor")
ax.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.5, color="xkcd:black", label="Detector")

m = 0
n = 0
for i in positions:
    for j in i:
        if np.fabs(j) < 1e-8:
           positions[m,n] = 0
        n = n + 1
    n = 0
    m = m + 1


positions_for_testing = positions[0:-1:2]

ax.plot([row[0] for row in positions_for_testing], [row[1] for row in positions_for_testing], [row[2] for row in positions_for_testing], ".")

world_limits = ax.get_w_lims()
ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))
plt.show()


data = pd.DataFrame()
data["position_x"] = [row[0] for row in positions_for_testing]
data["position_y"] = [row[1] for row in positions_for_testing]
data["position_z"] = [row[2] for row in positions_for_testing]

#data.to_csv("grid_positions.csv", index=False)




