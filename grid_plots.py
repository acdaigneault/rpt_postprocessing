import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})


# Extract data
#path_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/nomad/"
from matplotlib.cm import ScalarMappable

path_data = "/mnt/DATA/rpt_postprocessing/"
filename = "grid_counts.csv"
data_file = pd.read_csv(path_data + filename, sep=",")
data_type = "r"#

# Reactor dimensions
L = 0.3 # m
R = 0.1 # m

# Detector dimensions
r = 0.0381 # m
l = 0.0762 # m

# Position of the detector
FP = [0.2, 0, 0.075]
MP = [FP[0] + l/2, FP[1], FP[2]]

# Detector curve
detector_xz = patches.Rectangle((FP[0], FP[2]-r), 2*r, l,linestyle="--", linewidth=1, edgecolor='k', facecolor='none')
detector_yz = patches.Circle((FP[1], FP[2]), r, linestyle="--", linewidth=1, edgecolor='k', facecolor='none')
reactor = patches.Circle((0 ,0), R, linestyle="-", linewidth=1, edgecolor='k', facecolor='none')


# Error functions
def calculate_relative_error(calculated, measured):
    return np.fabs(calculated - measured)/measured*100

def calculate_absolute_error(calculated, measured):
    return np.fabs(calculated - measured)

# Function for plotting
def plotting_yz(data_list, save_name, X, Y): #(data_list, marker_size_list, detector, save_name, color, X, Y):
    fig, ax = plt.subplots()
    sizes = [1000, 600, 300, 100]

    zs = np.concatenate(data_list, axis=0)
    cmap = plt.get_cmap("Reds")
    norm = plt.Normalize(zs.min(), zs.max())

    for data, size in zip(data_list, sizes):
        for i, x, y in zip(data, X, Y):
            ax.scatter(x, y, c=[cmap(norm(i))], s=size, linewidths=0.25, edgecolors="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax)
    detector_yz = patches.Circle((FP[1], FP[2]), r, linestyle="--", linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(detector_yz)
    # ax.set_title("Décomptes pour plusieurs nombre itérations de Monte-Carlo à x = 0.")
    ax.set_xlabel("Position en y (m)")
    ax.set_ylabel("Position en z (m)")
    ax.set_xlim(-R, R) # y ou z
    ax.set_ylim(0, L) # y ou x
    ax.set_aspect("equal", "box")
    fig.set_size_inches(9, 6)
    #fig.savefig("/home/audrey/zpicture_presentation/" + save_name + ".png")
    plt.show()
    plt.close(fig)
    ax.clear()


def plotting_xz(data_list, save_name, X, Y):  # (data_list, marker_size_list, detector, save_name, color, X, Y):
    fig, ax = plt.subplots()
    sizes = [1000, 600, 300, 100]

    zs = np.concatenate(data_list, axis=0)
    cmap = plt.get_cmap("Blues")
    norm = plt.Normalize(zs.min(), zs.max())

    for data, size in zip(data_list, sizes):
        for i, x, y in zip(data, X, Y):
            ax.scatter(x, y, c=[cmap(norm(i))], s=size, linewidths=0.25, edgecolors="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax)

    detector_1d_side = plt.vlines(R, FP[2] - r, FP[2] + r, linestyle="--", linewidth=2, color="black")
    # ax.set_title("Décomptes pour plusieurs nombre itérations de Monte-Carlo à x = 0.")
    ax.set_xlabel("Position en x (m)")
    ax.set_ylabel("Position en z (m)")
    ax.set_xlim(-R, R)
    ax.set_ylim(0, L)
    ax.set_aspect("equal", "box")
    fig.set_size_inches(8, 5.3333)
    # fig.savefig("/home/audrey/zpicture_presentation/" + save_name + ".png")
    plt.show()
    plt.close(fig)
    ax.clear()

def plotting_xy(data_list, save_name, X, Y):  # (data_list, marker_size_list, detector, save_name, color, X, Y):
    fig, ax = plt.subplots()
    sizes = [1000, 600, 300, 100]

    zs = np.concatenate(data_list, axis=0)
    cmap = plt.get_cmap("Greens")
    norm = plt.Normalize(zs.min(), zs.max())

    for data, size in zip(data_list, sizes):
        for i, x, y in zip(data, X, Y):
            ax.scatter(x, y, c=[cmap(norm(i))], s=size, linewidths=0.25, edgecolors="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax)
    detector_1d_top = plt.vlines(R, FP[1] - r, FP[1] + r, linestyle="--", linewidth=2, color="black")
    reactor = patches.Circle((0, 0), R, linestyle="-", linewidth=1, edgecolor='k', facecolor='none')
    ax.add_patch(reactor)
    # ax.set_title("Décomptes pour plusieurs nombre itérations de Monte-Carlo à x = 0.")
    ax.set_xlabel("Position en x (m)")
    ax.set_ylabel("Position en y (m)")
    ax.set_xlim(-R, R)
    ax.set_ylim(-R, R)
    ax.set_aspect("equal", "box")
    fig.set_size_inches(6,6)
    # fig.savefig("/home/audrey/zpicture_presentation/" + save_name + ".png")
    plt.show()
    plt.close(fig)
    ax.clear()


plans = ["yz", "xz", "xy"]
constant_axes = ["x", "y", "z"]
constant_values = [0, 0, 0.0909091]
data_grid = [pd.DataFrame(columns=data_file.columns)] * len(constant_axes)
iterations = ["1000", "10000", "100000", "1000000"]
it_max = "10000000"



for ax, cte_value, i_grid in zip(constant_axes, constant_values, [0, 1, 2]):
    # Search positions with that constant value
    index_list = []
    for index, value in enumerate(data_file["particle_positions_" + ax]):
        if np.isclose(value, cte_value):
            index_list.append(index)

        if ax == "z" and np.isclose(value, 0.067272727272727):
            index_list.append(index)

    data_grid[i_grid] = data_file.loc[index_list, :].copy()
    counts_max_it = data_grid[i_grid]["counts_it" + it_max]
    data = []
    for it in iterations:
        if data_type == "relative":
            counts = data_grid[i_grid]["counts_it" + it]
            error = calculate_relative_error(counts, counts_max_it)
            data.append(error)
        elif data_type == "absolute":
            counts = data_grid[i_grid]["counts_it" + it]
            error = calculate_absolute_error(counts, counts_max_it)
            data.append(error)
        else:
            counts = data_grid[i_grid]["counts_it" + it]
            data.append(counts)

    save_name = data_type + "_" + plans[i_grid]

    X = data_grid[i_grid]["particle_positions_" + plans[i_grid][0]]
    Y = data_grid[i_grid]["particle_positions_" + plans[i_grid][1]]
    if plans[i_grid] == "yz":
        plotting_yz(data, save_name, X, Y)
    elif plans[i_grid] == "xz":
        plotting_xz(data, save_name, X, Y)
    else:
        plotting_xy(data, save_name, X, Y)

fig3d = plt.figure()
geo = plt.axes(projection="3d")

def data_for_cylinder(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Zd, Yd, Xd = data_for_cylinder(FP[2],FP[1],r,l)
reactor = geo.plot_surface(Xc, Yc, Zc, alpha=0.2, color="grey", label="Reactor")
detector = geo.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.5, color="xkcd:black", label="Detector")

for i_grid, color in enumerate(["red", "blue", "green"]):
    grid1 = geo.plot(data_grid[i_grid]["particle_positions_x"], data_grid[i_grid]["particle_positions_y"],
                     data_grid[i_grid]["particle_positions_z"], ".", markersize=10, color=color)


world_limits = geo.get_w_lims()
geo.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

plt.show()





"""

# Grid #1 : y and z at x = cte
x = 0


index_list = []
for index, x_value in enumerate(data_file["particle_positions_x"]): # List of y and z position at x = 0
    if np.isclose(x_value, x):
        index_list.append(index)

data_grid1 = data_file.loc[index_list, :].copy()
counts_err1000 = np.fabs(data_grid1["counts_it1000"] - data_grid1["counts_it10000000"])#/data_grid1["counts_it10000000"]
counts_err10000 = np.fabs(data_grid1["counts_it10000"] - data_grid1["counts_it10000000"])#/data_grid1["counts_it10000000"]
counts_err100000 = np.fabs(data_grid1["counts_it100000"] - data_grid1["counts_it10000000"])#/data_grid1["counts_it10000000"]
counts_err1000000 = np.fabs(data_grid1["counts_it1000000"] - data_grid1["counts_it10000000"])#/data_grid1["counts_it10000000"]

# Plot count's error on the grid
sizes = [1000, 600, 300, 100]
errors = [counts_err1000, counts_err10000, counts_err100000, counts_err1000000]
save = "error_it_1"
plotting(errors, sizes, detector_yz, save, "Reds", data_grid1["particle_positions_y"], data_grid1["particle_positions_z"])
# ax.set_title("Décomptes pour plusieurs nombre itérations de Monte-Carlo à x = 0.")
# ax.set_title("Différence relative des décomptes pour plusieurs nombre itérations de Monte-Carlo à x = 0.")



# Grid #2 : y and z at x = cte
y = 0

data_grid2 = pd.DataFrame(columns=data_file.columns)
index_list = []

for index, y_value in enumerate(data_file["particle_positions_y"]): # List of x and z position at y = 0
    if np.isclose(y_value, y):
        index_list.append(index)

data_grid2 = data_file.loc[index_list, :].copy()

counts_err1000 = np.fabs(data_grid2["counts_it1000"] - data_grid2["counts_it10000000"])#/data_grid2["counts_it10000000"]
counts_err10000 = np.fabs(data_grid2["counts_it10000"] - data_grid2["counts_it10000000"])#/data_grid2["counts_it10000000"]
counts_err100000 = np.fabs(data_grid2["counts_it100000"] - data_grid2["counts_it10000000"])#/data_grid2["counts_it10000000"]
counts_err1000000 = np.fabs(data_grid2["counts_it1000000"] - data_grid2["counts_it10000000"])#/data_grid2["counts_it10000000"]

zs = np.concatenate([counts_err1000, counts_err10000, counts_err100000, counts_err1000000], axis=0)
cmap = plt.get_cmap("Reds")
norm = plt.Normalize(zs.min(), zs.max())

# Plot count's error on the grid
sizes = [1000, 600, 300, 100]
errors = [counts_err1000, counts_err10000, counts_err100000, counts_err1000000]
save = "count_it_2"
color = "Blues"
#plotting(errors, sizes, detector_xz, save, color, data_grid2["particle_positions_x"], data_grid2["particle_positions_z"])


# Grid #3 : x and y at z = cte
z = 0.0909091

data_grid3 = pd.DataFrame(columns=data_file.columns)
index_list = []

for index, z_value in enumerate(data_file["particle_positions_z"]):
    if np.isclose(z_value, z):
        index_list.append(index)

data_grid3 = data_file.loc[index_list, :].copy()

counts_err1000 = np.fabs(data_grid3["counts_it1000"])# - data_grid3["counts_it10000000"])/data_grid3["counts_it10000000"]
counts_err10000 = np.fabs(data_grid3["counts_it10000"])# - data_grid3["counts_it10000000"])/data_grid3["counts_it10000000"]
counts_err100000 = np.fabs(data_grid3["counts_it100000"])# - data_grid3["counts_it10000000"])/data_grid3["counts_it10000000"]
counts_err1000000 = np.fabs(data_grid3["counts_it1000000"])# - data_grid3["counts_it10000000"])/data_grid3["counts_it10000000"]


# Plot count's error on the grid
sizes = [1000, 600, 300, 100]
errors = [counts_err1000, counts_err10000, counts_err100000, counts_err1000000]
save = "count_it_3"
color = "Greens"
#plotting(errors, sizes, reactor, save, color, data_grid3["particle_positions_x"], data_grid3["particle_positions_y"])


def data_for_cylinder(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

fig3d = plt.figure()
geo = plt.axes(projection="3d")

Xc, Yc, Zc = data_for_cylinder(0,0,R,L)
Zd, Yd, Xd = data_for_cylinder(FP[2],FP[1],r,l)
reactor = geo.plot_surface(Xc, Yc, Zc, alpha=0.2, color="grey", label="Reactor")
detector = geo.plot_surface(Xd+FP[0], Yd, Zd, alpha=0.5, color="xkcd:black", label="Detector")
grid1 = geo.plot(data_grid1["particle_positions_x"], data_grid1["particle_positions_y"], data_grid1["particle_positions_z"], ".", markersize=15, color="red")
grid2 = geo.plot(data_grid2["particle_positions_x"], data_grid2["particle_positions_y"], data_grid2["particle_positions_z"], ".", markersize=15, color="blue")
grid3 = geo.plot(data_grid3["particle_positions_x"], data_grid3["particle_positions_y"], data_grid3["particle_positions_z"], ".", markersize=15, color="green")


world_limits = geo.get_w_lims()
geo.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

plt.show()


counts_err1000 = np.fabs(data_file["counts_it1000"] - data_file["counts_it10000000"]) / data_file["counts_it10000000"]
counts_err10000 = np.fabs(data_file["counts_it10000"] - data_file["counts_it10000000"]) / data_file["counts_it10000000"]
counts_err100000 = np.fabs(data_file["counts_it100000"] - data_file["counts_it10000000"]) / data_file["counts_it10000000"]
counts_err1000000 = np.fabs(data_file["counts_it1000000"] - data_file["counts_it10000000"]) / data_file["counts_it10000000"]

print(4)

"""