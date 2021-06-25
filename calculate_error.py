import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


# Error type (relative or absolute)
error_type = "relative"

path_data = "/mnt/DATA/rpt_postprocessing/"
#filename = "grid_counts.csv"
filename = "grid_positions.csv"

data = pd.read_csv(path_data + filename, sep=",")
#iterations = ["1000", "10000", "100000", "1000000"]
iterations = ["1000", "10000", "100000", "1000000", "10000000"]
#n_runs = ["", "1", "2"]
n_runs = [""]

def calculate_relative_error(calculated, measured):
    return np.fabs(calculated - measured)/measured

def calculate_absolute_error(calculated, measured):
    return np.fabs(calculated - measured)

error_data = pd.DataFrame()
max_it = data["counts_it10000000"]
mean_vector = np.array([])
max_vector = np.array([])
min_vector = np.array([])

for nb in iterations:
    mean = np.array([])
    min = np.array([])
    max = np.array([])
    for run in n_runs:
        #it = data["counts_it" + nb + run]
        it = data["noisy_counts_it" + nb + run]
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
    min = np.min(min)
    min_vector = np.append(min_vector, min)
    max = np.max(max)
    max_vector = np.append(max_vector, max)
    print(f"{nb} itérations : max = {max}, min = {min}, moyenne = {mean}")

#plt.hist(error_data["relative_error_it1000000"],15)
fig, ax = plt.subplots()
it = [1000, 10000, 100000, 1000000, 10000000]
ax.semilogx(it, mean_vector, '.')
ax.semilogx(it, min_vector,'.')
ax.semilogx(it, max_vector,'.')
#ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)



plt.show()