import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


# Extract data
#path_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/nomad/"
from matplotlib.cm import ScalarMappable

path_data = "/mnt/DATA/rpt_postprocessing/"
filename = "grid_positions.csv"
data_file = pd.read_csv(path_data + filename, sep=",")
data_type = "relative"

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
global sizes
#sizes = [1000, 600, 300, 100]
sizes = [800, 250]

# Error functions
def calculate_relative_error(calculated, measured):
    return np.fabs(calculated - measured)/measured

def calculate_absolute_error(calculated, measured):
    return np.fabs(calculated - measured)

# Function for plotting
def plotting_yz(data_list, save_name, X, Y): #(data_list, marker_size_list, detector, save_name, color, X, Y):
    fig, ax = plt.subplots()

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
    fig.set_size_inches(9, 6)
    # fig.savefig("/home/audrey/zpicture_presentation/" + save_name + ".png")
    plt.show()
    plt.close(fig)
    ax.clear()

def plotting_xy(data_list, save_name, X, Y):  # (data_list, marker_size_list, detector, save_name, color, X, Y):
    fig, ax = plt.subplots()

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
    fig.set_size_inches(9, 6)
    # fig.savefig("/home/audrey/zpicture_presentation/" + save_name + ".png")
    plt.show()
    plt.close(fig)
    ax.clear()


plans = ["yz", "xz", "xy"]
constant_axes = ["x", "y", "z"]
constant_values = [0, 0, 0.0909091]
data_grid = [pd.DataFrame(columns=data_file.columns)] * len(constant_axes)
#iterations = ["10000", "100000", "1000000", "10000000"]
iterations = ["10000", "100000"]




for ax, cte_value, i_grid in zip(constant_axes, constant_values, [0, 1, 2]):
    # Search positions with that constant value
    index_list = []
    for index, value in enumerate(data_file["particle_positions_" + ax]):
        if np.isclose(value, cte_value):
            index_list.append(index)

        if ax == "z" and np.isclose(value, 0.067272727272727):
            index_list.append(index)

    data_grid[i_grid] = data_file.loc[index_list, :].copy()
    counts_max_it = data_grid[i_grid]["counts_it10000000"]
    data = []
    for it in iterations:
        if data_type == "relative":
            counts = data_grid[i_grid]["noisy_counts_it" + it]
            error = calculate_relative_error(counts, counts_max_it)
            data.append(error)
        elif data_type == "absolute":
            counts = data_grid[i_grid]["noisy_counts_it" + it]
            error = calculate_absolute_error(counts, counts_max_it)
            data.append(error)
        else:
            counts = data_grid[i_grid]["noisy_counts_it" + it]
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

