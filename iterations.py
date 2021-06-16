import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to data file and data filename
path_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/positions/"
it_file = "/positions_it.csv"

data = pd.read_csv(path_data + it_file, sep=";")

positions = [np.array([])]
skip_line = 0  # skip line
direction_set = []
for i in range(1, data.shape[0]):

    # Track what direction the particle is moving with booleans
    find_position_bool = [np.fabs(data["particle_positions_x"][i] - data["particle_positions_x"][i-1]) < 1e-6,
                          np.fabs(data["particle_positions_y"][i] - data["particle_positions_y"][i-1]) < 1e-6,
                          np.fabs(data["particle_positions_z"][i] - data["particle_positions_z"][i-1]) < 1e-6]


    if skip_line == 1:  # moving in a new direction (get the last position)
        if last_position == "x":
            positions[-1] = np.append(positions[-1], data["particle_positions_x"][i-2])
        elif last_position == "y":
            positions[-1] = np.append(positions[-1], data["particle_positions_y"][i-2])
        else:
            positions[-1] = np.append(positions[-1], data["particle_positions_z"][i-2])

        skip_line = 0
        positions.append(np.array([]))


    if find_position_bool == [0, 1, 1]: # moving in x direction
        positions[-1] = np.append(positions[-1], data["particle_positions_x"][i-1])
        last_position = "x"
    elif find_position_bool == [1, 0, 1]: # moving in y direction
        positions[-1] = np.append(positions[-1], data["particle_positions_y"][i-1])
        last_position = "y"
    elif find_position_bool == [1, 1, 0]: # moving in z direction
        positions[-1] = np.append(positions[-1], data["particle_positions_z"][i-1])
        last_position = "z"
    else: # not moving in the same direction anymore
        skip_line = 1

    if skip_line == 1:
        direction_set.append(last_position)

    # Last line
    if i + 1 == data.shape[0]:
        if last_position == "x":
            positions[-1] = np.append(positions[-1], data["particle_positions_x"][i])
        elif last_position == "y":                                                    
            positions[-1] = np.append(positions[-1], data["particle_positions_y"][i])
        else:                                                                         
            positions[-1] = np.append(positions[-1], data["particle_positions_z"][i])

        direction_set.append(last_position)



# Store length of data per set
n_data_set = len(positions)
n_set = []
for i in range(n_data_set):
    n_set.append(len(positions[i]))

# Get counts data for all iterations
counts = np.array([data["counts it = 1000"], data["counts it = 10000"],
                  data["counts it = 100000"], data["counts it = 1000000"]])

fig, ax = plt.subplots()

ax.plot(positions[0], counts[0][0:n_set[0]], label="it = 1000")
ax.plot(positions[0], counts[1][0:n_set[0]], label="it = 10000")
ax.plot(positions[0], counts[2][0:n_set[0]], label="it = 100000")
ax.plot(positions[0], counts[3][0:n_set[0]], label="it = 1000000")
ax.plot(positions[0], counts[3][0:n_set[0]], label="it = 10000000")
ax.set_xlabel(direction_set[0])
ax.set_ylabel("Count")
ax.legend()
plt.show()
fig.savefig(
    path_data + "set0.png")
plt.close(fig)
ax.clear()

for i in range(1, n_data_set):
    ax.plot(positions[i], counts[0][n_set[i-1]:n_set[i-1]+n_set[i]], label="it = 1000")
    ax.plot(positions[i], counts[1][n_set[i-1]:n_set[i-1]+n_set[i]], label="it = 10000")
    ax.plot(positions[i], counts[2][n_set[i-1]:n_set[i-1]+n_set[i]], label="it = 100000")
    ax.plot(positions[i], counts[3][n_set[i-1]:n_set[i-1]+n_set[i]], label="it = 1000000")
    ax.plot(positions[i], counts[3][n_set[i-1]:n_set[i-1]+n_set[i]], label="it = 10000000")
    ax.set_xlabel(direction_set[i])
    ax.set_ylabel("Count")
    ax.legend()
    plt.show()
    fig.savefig(path_data + "set" + str(i) + ".png")
    plt.close(fig)
    ax.clear()
