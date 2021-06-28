import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})



# Error type (relative or absolute)
error_type = "relative"

path_data = "/mnt/DATA/rpt_postprocessing/"
filename = "grid_positions.csv"
#1filename = "grid_counts.csv"

data = pd.read_csv(path_data + filename, sep=",")
#iterations = ["1000", "10000", "100000", "1000000"]
#iterations = ["4", "5"]
iterations = ["nomad_run2", "nomad_run3", "nomad_run5", "nomad_run6"]
#n_runs = ["", "1", "2"]
n_runs = [""]

def calculate_relative_error(calculated, measured):
    return np.fabs(calculated - measured)/measured*100

def calculate_absolute_error(calculated, measured):
    return np.fabs(calculated - measured)

error_data = pd.DataFrame()
max_it = data["counts_it10000000"]
mean_vector = np.array([])
max_vector = np.array([])
min_vector = np.array([])
sd_vector = np.array([])

for nb in iterations:
    mean = np.array([])
    min = np.array([])
    max = np.array([])
    for run in n_runs:
        #it = data["counts_it" + nb + run]
        #it = data["nomad_run" + nb]
        it = data[nb]
        if error_type == "relative":
            #error_data["relative_error_it" + nb + run] = calculate_relative_error(it, max_it)
            error_data["relative_error_it" + nb + run] = calculate_relative_error(it, max_it)
            mean = np.append(mean, np.mean(error_data["relative_error_it" + nb + run]))
            min = np.append(min, np.min(error_data["relative_error_it" + nb + run]))
            max = np.append(max, np.max(error_data["relative_error_it" + nb + run]))
        else:
            error_data["absolute_error_it" + nb + run] = calculate_absolute_error(it, max_it)
            mean = np.append(mean, np.mean(error_data["absolute_error_it" + nb + run]))
            min = np.append(min, np.min(error_data["absolute_error_it" + nb + run]))
            max = np.append(max, np.max(error_data["absolute_error_it" + nb + run]))

    sd = np.sqrt((error_data["relative_error_it" + nb + run]**2).sum()/error_data["relative_error_it" + nb + run].shape[0])
    sd_vector = np.append(sd_vector, sd)
    mean = np.mean(mean)
    mean_vector = np.append(mean_vector, mean)
    min = np.min(min)
    min_vector = np.append(min_vector, min)
    max = np.max(max)
    max_vector = np.append(max_vector, max)
    print(f"{nb} itérations : max = {max}, min = {min}, moyenne = {mean}, écart-type = {sd}, 95% IDC = {1.96*sd}")

#plt.hist(error_data["relative_error_it1000000"],15)
fig, ax = plt.subplots()
it = [0.975*10000, 1.025*10000, 0.975*100000, 1.025*100000]
ax.semilogx(np.take(it,[0,2]), np.take(mean_vector,[0,2]), '.', label="Erreur relative moyenne", color="black")
#ax.loglog(it, min_vector,'.', label="Erreur relative minimale", color="tab:green")
ax.semilogx(it, max_vector,'.', label="Erreur relative maximale", color="tab:blue")
#ax.set_title("Erreur relative des résultats de décomptes avec les positions bruitées")
ax.set_xlabel("Nombre d'itérations de Monte-Carlo")
ax.set_ylabel(r"Erreur relative (\%)")
fig.set_size_inches(6, 4)
plotline, capline, barlinecol = ax.errorbar(it, mean_vector, yerr=1.96*sd_vector, capsize=0, ls="None", color="black", lolims=True, label=r"Intervalle de confiance de 95\%")
capline[0].set_marker("_")
ax.legend()

#fig2, ax2 = plt.subplots()
#bleh = np.array(error_data["relative_error_it100000"]), np.array(error_data["relative_error_it1000001"]), np.array(error_data["relative_error_it1000002"])
#ax2.hist(bleh, 15)

#fig.savefig("/home/audrey/image_presentation/error/error_tune_idc" + ".png", dpi=200)

plt.show()