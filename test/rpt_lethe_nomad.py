"""
Name   : rpt_lethe_nomad.py
Author : Audrey Collard-Daigneault
Date   : 17-06-2021
Desc   : This code is a mega Python code to run the rpt_3d lethe application with the NOMAD software for
         blackbox optimization
"""

import os
import sys

# Path to files/application
# path = "/mnt/DATA/rpt_postprocessing/lethe_nomad/"
path = "./"  # Current directory

path_to_rpt_3d = "/home/audrey/work/lethe/build/applications/rpt_3d/rpt_3d"

# Parameter filenames
initial_lethe_parameter_file = "rpt_parameters.prm"
new_lethe_parameter_file = "rpt_new_parameters.prm"
nomad_parameter_file = "param_nomad.txt"

# List of parameters to tune
parameters_to_tune = ["dead time", "activity", "attenuation coefficient reactor"]
n_parameters_to_tune = len(parameters_to_tune)

# Get new calculated parameters
tmpfile = sys.argv[1]  # Temporary file for new values
with open(tmpfile, "r") as values:
    new_values = values.read().split()

# Search to modify .prm new calculated tuned parameters
with open(path + initial_lethe_parameter_file, "r") as file:  # Read .prm file with split lines
    filestr = file.read().split("\n")

for line in range(0, len(filestr)):
    if "=" in filestr[line]:
        last_value = filestr[line].split("=")[1]  # get the last value of the line

        for i in range(0, n_parameters_to_tune):
            if parameters_to_tune[i] in filestr[line]:
                filestr[line] = filestr[line].replace(last_value, " " + new_values[i])

with open(path + new_lethe_parameter_file, "w") as file:  # Write .prm with new parameters
    file.write("\n".join(filestr))

# Call rpt_3d executable
os.system(path_to_rpt_3d + " " + path + new_lethe_parameter_file)
