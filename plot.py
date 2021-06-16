import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Extract data
path_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/nomad/"
#filename = "positions_counts.csv"
filename = "set1.csv"
data = pd.read_csv(path_data + filename, sep=";")

n_data_per_set = 1
n_set = int(data.shape[0]/n_data_per_set)

angle_distance = [np.array(data["angle_distance"].iloc[0:n_data_per_set])]
count = [np.array(data["count"].iloc[0:n_data_per_set])]

for i in range(n_data_per_set, data.shape[0], n_data_per_set):
    angle_distance.append(np.array(data["angle_distance"].iloc[i:i+n_data_per_set]))
    count.append(np.array(data["count"].iloc[i:i+n_data_per_set]))

plt.plot(angle_distance, count,'.')

"""
# Plot initialization
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)

fig2 = plt.figure()
ax3 = fig2.add_subplot(1, 3, 1)
ax4 = fig2.add_subplot(1, 3, 2)
ax5 = fig2.add_subplot(1, 3, 3)

fig3 = plt.figure()
ax6 = fig3.add_subplot(2, 2, 1)
ax7 = fig3.add_subplot(2, 2, 2)
ax8 = fig3.add_subplot(2, 2, 3)
ax9 = fig3.add_subplot(2, 2, 4)

# Plot of positions with constant distance & variable angle
ax.plot(angle_distance[0], count[0], ".", color="red")
ax1.plot(angle_distance[1], count[1], ".", color="red")
ax2.plot(angle_distance[2], count[2], ".", color="red")

fig.suptitle("Constant distance with variable angle")
ax.set_ylabel("Count")
ax1.set_xlabel("Angle")

# Plot of positions with constant angle & variable distance
ax3.plot(angle_distance[3], count[3], ".", color="red")
ax4.plot(angle_distance[4], count[4], ".", color="red")
ax5.plot(angle_distance[5], count[5], ".", color="red")

fig2.suptitle("Constant angle with variable distance")
ax3.set_ylabel("Count")
ax4.set_xlabel("Distance")

# Plot of positions with one variable direction
ax6.plot(angle_distance[6], count[6], ".", color="red")
ax7.plot(angle_distance[7], count[7], ".", color="red")
ax8.plot(angle_distance[8], count[8], ".", color="red")
ax9.plot(angle_distance[9], count[9], ".", color="red")

fig3.suptitle("One variable direction")
ax6.set_ylabel("Count")
ax7.set_xlabel("Distance")

"""
plt.show()