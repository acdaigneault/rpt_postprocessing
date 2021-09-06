import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
global save_name
it = "1000000"
save_name = "/home/audrey/image_presentation/symmetry_percentage_" + it
image_format = ".png"

# Names of the columns
#column_names = ["counts_it10000", "counts_it100000", "counts_it1000000"]
#column_names = ["counts_it1000000", "nomad_run1"]
column_names = ["nomad_run5", "nomad_run11"]

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
plans = ["yz"]
constant_axes = ["x", "y", "z"]
constant_axes = ["x"]
constant_values = [0, 0, 0.0909091] # Constant value position
data_grid = [pd.DataFrame(columns=data_file.columns)] * len(constant_axes)
reference_data = "counts_it" + it # data to evaluate error
column_names = [reference_data]
ref_data = data_file["counts_it10000000"] # data to evaluate error

pos = []
diff = []
j_list = []
for i in range(0, data_file.shape[0]):
    for j in range(0, data_file.shape[0]):
        if (abs(data_file["particle_positions_x"][i]) < 1e-6 and data_file["particle_positions_x"][j] < 1e-6  and
            abs(data_file["particle_positions_z"][i] - data_file["particle_positions_z"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i]) > 1e-6 and
            j is not i and
            i not in j_list):
            pos.append([data_file["particle_positions_x"][i],
                                  data_file["particle_positions_y"][i],
                                  data_file["particle_positions_z"][i]])
            diff = np.append(diff, abs(data_file[reference_data  + ""][i] -
                                       data_file[reference_data + ""][j]))
            j_list.append(j)
            #print(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j])
            #print(pos[-1],diff[-1])

print(f"Max : {np.max(diff)}, min {np.min(diff)}, moyenne {np.mean(diff)}")
a = 1


"""
pos1 = []
diff1 = []
j_list1 = []
for i in range(0, data_file.shape[0]):
    for j in range(0, data_file.shape[0]):
        if (abs(data_file["particle_positions_x"][i]) < 1e-6 and data_file["particle_positions_x"][j] < 1e-6  and
            abs(data_file["particle_positions_z"][i] - data_file["particle_positions_z"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i]) > 1e-6 and
            j is not i and
            i not in j_list1):
            pos1.append([data_file["particle_positions_x"][i],
                                  data_file["particle_positions_y"][i],
                                  data_file["particle_positions_z"][i]])
            diff1 = np.append(diff1, abs(data_file[reference_data + "1"][i] -
                                       data_file[reference_data][j])/((ref_data[i]+ref_data[j])/2)*100)
            j_list1.append(j)
            #print(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j])
            #print(pos1[-1],diff1[-1])


pos2 = []
diff2 = []
j_list2 = []
for i in range(0, data_file.shape[0]):
    for j in range(0, data_file.shape[0]):
        if (abs(data_file["particle_positions_x"][i]) < 1e-6 and data_file["particle_positions_x"][j] < 1e-6  and
            abs(data_file["particle_positions_z"][i] - data_file["particle_positions_z"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j]) < 1e-6 and
            abs(data_file["particle_positions_y"][i]) > 1e-6 and
            j is not i and
            i not in j_list2):
            pos2.append([data_file["particle_positions_x"][i],
                                  data_file["particle_positions_y"][i],
                                  data_file["particle_positions_z"][i]])
            diff2 = np.append(diff2, abs(data_file[reference_data + "2"][i] -
                                       data_file[reference_data][j])/((ref_data[i]+ref_data[j])/2)*100)
            j_list2.append(j)
            #print(data_file["particle_positions_y"][i] + data_file["particle_positions_y"][j])
            #print(pos2[-1],diff2[-1])
"""
# Sizes for scatter (only for 1 to 4 columns max)
global sizes
a = 3
if a == 1:
    sizes = [900]
elif a == 2:
    sizes = [800, 300]
elif a == 3:
    sizes = [900, 500, 200]
elif a == 4:
    sizes = [1000, 600, 300, 100]


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
    divider = make_axes_locatable(ax)

    # Show the detector face prior plan
    if plan == "yz":
        detector_yz = patches.Circle((FP[1], FP[2]), r, linestyle="--", linewidth=1, edgecolor='k',
                                     facecolor='none')
        ax.add_patch(detector_yz)
        ax.set_xlabel("Position en y (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(0, R)
        ax.set_ylim(0, L)
        fig.set_size_inches(3, 5.5)
        cax = divider.append_axes("right", size="18%", pad=0.05)

    elif plan == "xz":
        detector_1d_side = plt.vlines(R, FP[2] - r, FP[2] + r, linestyle="--", linewidth=2, color="black")
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en z (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(0, L)
        fig.set_size_inches(4.5, 5.5)
        cax = divider.append_axes("right", size="9%", pad=0.05)

    elif plan == "xy":
        detector_1d_top = plt.vlines(R, FP[1] - r, FP[1] + r, linestyle="--", linewidth=2, color="black")
        reactor = patches.Circle((0, 0), R, linestyle="-", linewidth=1, edgecolor='k', facecolor='none')
        ax.add_patch(reactor)
        ax.set_xlabel("Position en x (m)")
        ax.set_ylabel("Position en y (m)")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        fig.set_size_inches(6, 5.5)
        cax = divider.append_axes("right", size="6.5%", pad=0.05)

    # If there's title
    if title != 0:
        ax.set_title(title)

    cb = fig.colorbar(sm, ax=ax, cax=cax)
    cb.ax.set_title(r'$\%$')

    # If Error in %, add % sign to the color bar
    if code == "percentage":
        cb.ax.set_title(r'$\%$')
    elif code == "counts":
        cb.ax.set_title("Nombre de photons", size=10)

    # Set the equal scale and save figure
    ax.set_aspect("equal", "box")
    #fig.savefig(save_name + image_format, dpi=200)
   # plt.close(fig)
    #ax.clear()
    plt.show()


def plotting_yz(data_list, save_name, X, Y, title, code=0):
    color = "Reds"
    plan = "yz"
    plotting(data_list, save_name, X, Y, color, code, plan, title)


def plotting_xz(data_list, save_name, X, Y, title, code=0):  #
    color = "Blues"
    plan = "xz"
    plotting(data_list, save_name, X, Y, color, code, plan, title)


def plotting_xy(data_list, save_name, X, Y, title, code=0):
    color = "Greens"
    plan = "xy"
    plotting(data_list, save_name, X, Y, color, code, plan, title)

X = np.array(pos)[:,1]
Y = np.array(pos)[:,2]
plotting_yz([pd.Series(diff)], save_name, X, Y, 0, code=0)

