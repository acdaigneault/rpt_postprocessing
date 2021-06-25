import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Extract data
#path_data = "C:/Users/Acdai/OneDrive - polymtl.ca/Polytechnique/Session E2021/GCH8392 - Projet individuel de génie chimique/Data/nomad/"
path_data = "/mnt/DATA/rpt_postprocessing/positions/"
filename = "positions_counts0.csv"

data = pd.read_csv(path_data + filename, sep=";")

n_data_per_set = 25
n_set = int(data.shape[0]/n_data_per_set)

angle_distance = [np.array(data["angle_distance"].iloc[0:n_data_per_set])]
count_exp = [np.array(data["count_exp"].iloc[0:n_data_per_set])]
count_tuning = [np.array(data["count_tuning"].iloc[0:n_data_per_set])]
noisy_count = [np.array(data["noisy_count"].iloc[0:n_data_per_set])]
noisy_count_tuning = [np.array(data["noisy_count_tuning"].iloc[0:n_data_per_set])]



for i in range(n_data_per_set, data.shape[0], n_data_per_set):
    angle_distance.append(np.array(data["angle_distance"].iloc[i:i+n_data_per_set]))
    count_exp.append(np.array(data["count_exp"].iloc[i:i+n_data_per_set]))
    count_tuning.append(np.array(data["count_tuning"].iloc[i:i+n_data_per_set]))
    noisy_count.append(np.array(data["noisy_count"].iloc[i:i+n_data_per_set]))
    noisy_count_tuning.append(np.array(data["noisy_count_tuning"].iloc[i:i+n_data_per_set]))


# Plot initialization
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax1 = fig.add_subplot(1, 3, 2)
ax2 = fig.add_subplot(1, 3, 3)

fig1 = plt.figure()
ax3 = fig1.add_subplot(1, 2, 1)
ax4 = fig1.add_subplot(1, 2, 2)
"""
# Plot of positions with constant distance & variable angle
ax.plot(angle_distance[0], count_exp[0], ".", color="black", label="Décomptes expérimentaux artificiels")
ax.plot(angle_distance[0], count_tuning[0], ".", color="crimson", label="Décomptes obtenus avec les paramètres ajustés")
ax.plot(angle_distance[0], noisy_count[0], ".", color="gray", label="Décomptes expérimentaux artificiels bruités")
ax.plot(angle_distance[0], noisy_count_tuning[0], ".", color="royalblue", label="Décomptes obtenus avec les paramètres ajustés bruités")

ax1.plot(angle_distance[1], count_exp[1], ".", color="black")
ax1.plot(angle_distance[1], count_tuning[1], ".", color="crimson")
ax1.plot(angle_distance[1], noisy_count[1], ".", color="gray")
ax1.plot(angle_distance[1], noisy_count_tuning[1], ".", color="royalblue")

ax2.plot(angle_distance[2], count_exp[2], ".", color="black")
ax2.plot(angle_distance[2], count_tuning[2], ".", color="crimson")
ax2.plot(angle_distance[2], noisy_count[2], ".", color="gray")
ax2.plot(angle_distance[2], noisy_count_tuning[2], ".", color="royalblue")


fig.suptitle("Particules ayant une distance constante")
ax.set_title("Trajectoire 1 : distance & z = cte")
ax.set_ylabel("Count")
ax.set_xlabel("Angle")
ax1.set_title("Trajectoire 2 : distance & y = cte")
ax1.set_ylabel("Count")
ax1.set_xlabel("Angle")
ax2.set_title("Trajectoire 3 : distance = cte & y = z")
ax2.set_ylabel("Count")
ax2.set_xlabel("Angle")
fig.legend()

# Plot of positions with constant angle & variable distance
ax3.plot(angle_distance[3], count_exp[3], ".", color="black", label="Décomptes expérimentaux artificiels")
ax3.plot(angle_distance[3], count_tuning[3], ".", color="crimson", label="Décomptes obtenus avec les paramètres ajustés")
ax3.plot(angle_distance[3], noisy_count[3], ".", color="gray", label="Décomptes expérimentaux artificiels bruités")
ax3.plot(angle_distance[3], noisy_count_tuning[3], ".", color="royalblue", label="Décomptes obtenus avec les paramètres ajustés bruités")



ax4.plot(angle_distance[4], count_exp[4], ".", color="black")
ax4.plot(angle_distance[4], count_tuning[4], ".", color="crimson")
ax4.plot(angle_distance[4], noisy_count[4], ".", color="gray")
ax4.plot(angle_distance[4], noisy_count_tuning[4], ".", color="royalblue")


fig1.suptitle("Particules ayant un angle constant")
ax3.set_title("Trajectoire 4 : angle = cte")
ax3.set_ylabel("Count")
ax3.set_xlabel("Distance")
ax4.set_title("Trajectoire 5 : angle & y, z = cte")
ax4.set_ylabel("Count")
ax4.set_xlabel("Distance")
fig1.legend()

ax.set_box_aspect(1)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax4.set_box_aspect(1)

"""

### Calculate error

i = 0
tuning_error = np.fabs(np.concatenate(count_exp) - np.concatenate(count_tuning))/np.concatenate(count_exp)
noisy_tuning_error = np.fabs(np.concatenate(count_exp) - np.concatenate(noisy_count_tuning))/np.concatenate(count_exp)


fig2 = plt.figure()
ax5 = fig2.add_subplot(1, 1, 1)
ax5.plot(np.concatenate(count_exp), tuning_error,".", color="crimson", label="Erreur des résultats des données artificielles")
ax5.plot(np.concatenate(count_exp), noisy_tuning_error, "." ,color="royalblue", label="Erreur des résultats des données artificielles bruitées")
ax5.set_xlabel("Décompte expérimental artificiel")
ax5.set_ylabel("Erreur relative")
ax5.legend(loc="upper left")
#ax5.set_xlim(0,25000)
#ax5.set_ylim(0,0.09)
fig2.set_size_inches(10, 6)




# Cost function
f = 0

for i in range(0,data.shape[0]):
    f = f + ((data["noisy_count_tuning"][i] - data["count_exp"][i])/(data["noisy_count_tuning"][i] + data["count_exp"][i]))**2

print(f)
# Least-squares approach


fig3 = plt.figure()
ax6 = fig3.add_subplot(1, 1, 1)
tuning_error0 = np.fabs(np.concatenate(count_exp) - data["count_tuning0"]) /np.concatenate(count_exp)
noisy_tuning_error0 = np.fabs(np.concatenate(count_exp) - data["noisy_count_tuning0"]) /np.concatenate(count_exp)
ax6.plot(np.concatenate(count_exp), tuning_error,".", color="crimson", label="Erreur des résultats des données artificielles (x0 + 10%)")
ax6.plot(np.concatenate(count_exp), noisy_tuning_error, "." ,color="crimson")
ax6.plot(np.concatenate(count_exp), tuning_error0,".", color="royalblue", label="Erreur des résultats des données artificielles (x0 - 10%)")
ax6.plot(np.concatenate(count_exp), noisy_tuning_error0, "." ,color="royalblue")
ax6.set_xlabel("Décompte expérimental artificiel")
ax6.set_ylabel("Erreur absolue")
ax6.legend(loc="upper left")
#ax5.set_xlim(0,25000)
#ax5.set_ylim(0,0.09)
fig3.set_size_inches(10, 6)
plt.show()