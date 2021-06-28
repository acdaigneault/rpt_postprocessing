import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable


# Set LaTex font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Extract data
path_data = "/mnt/DATA/rpt_postprocessing/"
filename = "grid_positions.csv"
#filename = "grid_counts.csv"
data_file = pd.read_csv(path_data + filename, sep=",")

# Error type or counts
data_type = "relative"

# Path to save figures and format
global save_path
save_path = "/home/audrey/image_presentation/nomad/costfunc1it_"
image_format = ".png"

# Names of the columns
#column_names = ["counts_it10000", "counts_it100000", "counts_it1000000"]
#column_names = ["counts_it1000000", "nomad_run1"]
column_names = ["nomad_run3", "nomad_run5"]

# Reactor dimensions
L = 0.3 # m
R = 0.1 # m

# Detector dimensions
r = 0.0381 # m
l = 0.0762 # m

# Position of the detector
FP = [0.2, 0, 0.075]
MP = [FP[0] + l/2, FP[1], FP[2]]

# Grids information
plans = ["yz", "xz", "xy"] # plans to plot
constant_axes = ["x", "y", "z"]
constant_values = [0, 0, 0.0909091] # Constant value position
data_grid = [pd.DataFrame(columns=data_file.columns)] * len(constant_axes)
reference_data = "counts_it10000000" # data to evaluate error







# Sizes for scatter (only for 1 to 4 columns max)
global sizes

if len(column_names) == 1:
    sizes = [900]
elif len(column_names) == 2:
    sizes = [800, 300]
elif len(column_names) == 3:
    sizes = [900, 500, 200]
elif len(column_names) == 4:
    sizes = [1000, 600, 300, 100]

# Error functions
def calculate_relative_error(calculated, measured):
    return np.fabs(calculated - measured)/measured*100

def calculate_absolute_error(calculated, measured):
    return np.fabs(calculated - measured)

# Function for plotting
def plotting(data_list, save_name, X, Y, color, code, plan, title):
    fig, ax = plt.subplots()

    # Allow to scale the color bar to all data in plot
    zs = np.concatenate(data_list, axis=0)
    cmap = plt.get_cmap(color)
    norm = plt.Normalize(zs.min(), zs.max())

    for data, size in zip(data_list, sizes):
        for i, x, y in zip(data, X, Y):
            ax.scatter(x, y, c=[cmap(norm(i))], s=size, linewidths=0.25, edgecolors="black")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax)

    # If Error in %, add % sign to the color bar
    if code == 1:
        cb.ax.set_title(r'$\%$')

    # Show the detector face prior plan
    if plan == "yz":
        detector_yz = patches.Circle((FP[1], FP[2]), r, linestyle="--", linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(detector_yz)
        ax.set_xlabel("Position en y (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(0, L)
        fig.set_size_inches(8, 5.333)
    elif plan == "xz":
        detector_1d_side = plt.vlines(R, FP[2] - r, FP[2] + r, linestyle="--", linewidth=2, color="black")
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(0, L)
        fig.set_size_inches(8, 5.333)
    elif plan == "xy":
        detector_1d_top = plt.vlines(R, FP[1] - r, FP[1] + r, linestyle="--", linewidth=2, color="black")
        reactor = patches.Circle((0, 0), R, linestyle="-", linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(reactor)
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en y (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        fig.set_size_inches(7, 7)

    # If there's title
    if title != 0:
        ax.set_title(title)

    # Set the equal scale and save figure
    ax.set_aspect("equal", "box")
    fig.savefig(save_path + save_name + image_format, dpi=200)
    plt.close(fig)
    ax.clear()
    plt.show()

def plotting_yz(data_list, save_name, X, Y, title, code=0):
    color = "Reds"
    plan = "yz"
    plotting(data_list, save_name, X, Y, color, code, plan, title)


def plotting_xz(data_list, save_name, X, Y, title, code=0):  #
    color = "Blues"
    plan = "xz"
    plotting(data_list, save_name, X, Y, color, code, plan, title)

def plotting_xy(data_list, save_name,  X, Y, title, code=0):
    color = "Greens"
    plan = "xy"
    plotting(data_list, save_name, X, Y, color, code, plan, title)



for ax, cte_value, i_grid in zip(constant_axes, constant_values, [0, 1, 2]):
    # Search positions with that constant value
    index_list = []
    for index, value in enumerate(data_file["particle_positions_" + ax]):
        if np.isclose(value, cte_value):
            index_list.append(index)

        if ax == "z" and np.isclose(value, 0.067272727272727): # Extra z = cte
            index_list.append(index)

    data_grid[i_grid] = data_file.loc[index_list, :].copy()
    counts_max_it = data_grid[i_grid][reference_data]
    data = []
    for column_name in column_names:
        if data_type == "relative":
            counts = data_grid[i_grid][column_name]
            error = calculate_relative_error(counts, counts_max_it)
            data.append(error)
            percentage = 1
        elif data_type == "absolute":
            counts = data_grid[i_grid][column_name]
            error = calculate_absolute_error(counts, counts_max_it)
            data.append(error)
            percentage = 0
        else:
            counts = data_grid[i_grid][column_name]
            data.append(counts)
            percentage = 0

    save_name = data_type + "_" + plans[i_grid]
    title = 0


    X = data_grid[i_grid]["particle_positions_" + plans[i_grid][0]]
    Y = data_grid[i_grid]["particle_positions_" + plans[i_grid][1]]
    if plans[i_grid] == "yz":
        plotting_yz(data, save_name, X, Y, title, percentage)
    elif plans[i_grid] == "xz":
        plotting_xz(data, save_name, X, Y, title, percentage)
    else:
        plotting_xy(data, save_name, X, Y, title, percentage)

"""# Positions in reactor
fig3d = plt.figure()
geo =fig3d.add_subplot(projection="3d")

x = [-R,R,R,-R]
y = [0,0,0,0]
z = [0,0,L,L]
blue = [list(zip(x,y,z))]
rect = Poly3DCollection(blue, alpha=0.3, color="blue")
geo.add_collection3d(rect)

y = [-R,R,R,-R]
x = [0,0,0,0]
z = [0,0,L,L]
red = [list(zip(x,y,z))]
rect = Poly3DCollection(red, alpha=0.3, color="red")
geo.add_collection3d(rect)

theta = np.linspace(0, 2*np.pi, 100)
y = R*np.cos(theta)
x = R*np.sin(theta)
z = 0.0909091*np.ones(100)
green = [list(zip(x,y,z))]
circle = Poly3DCollection(green, alpha=0.3, color="green")
geo.add_collection3d(circle)

theta = np.linspace(0, 2*np.pi, 100)
y = R*np.cos(theta)
x = R*np.sin(theta)
z = 0.067272727272727*np.ones(100)
green = [list(zip(x,y,z))]
circle = Poly3DCollection(green, alpha=0.3, color="green")
geo.add_collection3d(circle)




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


geo.set_xlim(-R,FP[0]+l)
geo.set_ylim(-R,R)
geo.set_zlim(0,L)
geo.set_xlabel("Position en x (m)")
geo.set_ylabel("Position en y (m)")
geo.set_zlabel("Position en z (m)")
world_limits = geo.get_w_lims()
geo.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))

plt.show()
"""
