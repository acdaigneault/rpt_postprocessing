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
iterations = ["counts_it1000", "counts_it10000", "counts_it100000", "counts_it1000000"]
##iterations = ["4", "5"]
#iterations = ["nomad_run3", "nomad_run5", "nomad_run9","nomad_run11"]
#1iterations = ["noisy_counts_it10000000"]
n_runs = ["", "1", "2"]
#n_runs = [""]

def calculate_relative_error(calculated, measured):
    return np.abs(calculated - measured)/measured*100

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
        it = data[nb+run]
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

    mean = np.mean(mean)
    mean_vector = np.append(mean_vector, mean)
    sd = np.sqrt((np.sum((error_data["relative_error_it" + nb + run] - mean)**2))/error_data["relative_error_it" + nb + run].shape[0])
    sd_vector = np.append(sd_vector, sd)
    min = np.min(min)
    min_vector = np.append(min_vector, min)
    max = np.max(max)
    max_vector = np.append(max_vector, max)
    f = 0
    for i in range(0, data.shape[0]):
        f = f + ((data[nb][i] - data["noisy_counts_it10000000"][i])/(data[nb][i] + data["noisy_counts_it10000000"][i]))**2
    function = np.sum(((data[nb] - max_it)/(data[nb] + max_it))**2)
    print(f"{nb} itérations :  moyenne = {mean}, écart-type = {sd}, f = {f}, 95% IDC = {1.96*sd}, max = {max},")



"""fig0, ax0 = plt.subplots(2, 2)
ax0[0, 0].hist(error_data["relative_error_it" + iterations[0]],10)
ax0[0, 1].hist(error_data["relative_error_it" + iterations[1]],10)
ax0[1, 0].hist(error_data["relative_error_it" + iterations[2]],10)
ax0[1, 1].hist(error_data["relative_error_it" + iterations[3]],10)

plt.show()"""




fig, ax = plt.subplots()


if iterations[0] == "counts_it1000":
    sd_vector = 1.96*np.array([0.9703, 0.6283, 0.2036, 0.0334])
    it = [1000, 10000, 100000, 1000000]
    ax.semilogx(it, mean_vector, '.', label="Erreur relative moyenne",
                color="black")
    ax.semilogx(it, max_vector, '.', label="Erreur relative maximale",
                color="tab:blue")
    plotline, capline, barlinecol = ax.errorbar(it, mean_vector, yerr=sd_vector, capsize=0, lolims=True, ls="None",
                                                color="black", label=r"Intervalle de confiance de 95\%")
    capline[0].set_marker("_")
    #ax.axvline(x=4e4, ymin=np.log10(2.15), ymax=np.log10(2.36), ls="--", color="black", linewidth="1")
    #ax.axhline(y=0.26, xmin=np.log10(3), xmax=np.log10(3.4), ls="--", color="black", linewidth="1")
    #plt.text(4.6e4, 0.2125, r"-0.488", fontsize="10" )

    #ax.axvline(x=4e4, ymin=np.log10(4), ymax=np.log10(4.38), ls="--", color="tab:blue", linewidth="1")
    #ax.axhline(y=1.15, xmin=np.log10(3), xmax=np.log10(3.4), ls="--", color="tab:blue", linewidth="1")
    #plt.text(4.6e4, 0.955, r"-0.491", fontsize="10", color="tab:blue" )



else:
    it = [0.975 * 10000, 0.975 * 100000, 1.025 * 10000, 1.025 * 100000]
    ax.semilogx(np.take(it,[0,1]), np.take(mean_vector,[0,1]), '.', label=r"Erreur relative moyenne $P_{1}$", color="black")
    ax.semilogx(np.take(it,[2,3]), np.take(mean_vector,[2,3]), '.', label=r"Erreur relative moyenne $P_{2}$", color="tab:blue")
    ax.semilogx(np.take(it,[0,1]), np.take(min_vector,[0,1]), '.', color="grey")
    ax.semilogx(np.take(it,[2,3]), np.take(min_vector,[2,3]), '.', color="skyblue")
    ax.semilogx(np.take(it,[0,1]), np.take(max_vector,[0,1]),'.', label=r"Erreur relative maximale $P_{1}$", color="grey")
    ax.semilogx(np.take(it,[2,3]), np.take(max_vector,[2,3]),'.', label=r"Erreur relative maximale $P_{2}$", color="skyblue")
    #ax.set_title("Erreur relative des résultats de décomptes avec les positions bruitées")
    plotline, capline, barlinecol = ax.errorbar(it, mean_vector, yerr=1.96*sd_vector, capsize=0, lolims=True, ls="None",
                                                color="black", label=r"Intervalle de confiance de 95\%")
    capline[0].set_marker("_")


ax.set_xlabel("Nombre d'itérations de Monte-Carlo")
ax.set_ylabel(r"Erreur relative (\%)")
fig.set_size_inches(6, 4)

ax.legend()  #loc="upper center")
#plt.grid(color="grey", ls="--", linewidth="0.5", axis="y", which="both")

fig.savefig("/home/audrey/image_presentation/error_noabs_idc_" + ".png", dpi=200)

plt.show()