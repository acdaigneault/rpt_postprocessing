import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Path to data file and data filename
path_data = "."
counts_file = "/grid_positions.csv"

data = pd.read_csv(path_data + counts_file, sep=",")

"""
# Create a noisy count serie
noise = pd.Series(np.random.normal(0, 5, data.shape[0]))  # Normal distribution, mean = 0, sd = 5
noisy_count = pd.Series(data["count_exp"] + noise)

# Add noisy counts column to current csv
data["noisy_count"] = noisy_count # data.assign(noisy_count.values)
"""

# Generate noise
uncertainty = 0.002 # m

#noise = pd.Series(np.random.normal(0, uncertainty/3, 150))  # Normal distribution
noise = np.random.normal(0, uncertainty, 450)  # Normal distribution


positions = ["x", "y", "z"]
j = 0
for i in positions:
    noisy_count = data["particle_positions_" + i].add(noise[j:j+150], fill_value=0)
    data["noisy_particle_positions_" + i] = noisy_count
    j = j + 150


lol1 = pd.Series(noise[0:150])
lol2 = pd.Series(noise[150:300])
lol3 = pd.Series(noise[300:450])

boe= 1
#print(data["position_" + "y"] + noise[150:300])

data.to_csv(path_data + counts_file, index=False)