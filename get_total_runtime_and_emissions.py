"""
Add up the running time and emissions of all the experiments.
"""

# STD
import os

# EXT
import pandas as pd

# PROJECT
from src.constants import EMISSION_DIR


if __name__ == "__main__":
    # Get all the experiment emissions directories
    experiment_dirs = os.listdir(EMISSION_DIR)

    # Loop through directories, extract time and emissions
    total_time, total_emissions, total_kWH = 0, 0, 0

    for dir in experiment_dirs:
        try:
            data = pd.read_csv(os.path.join(EMISSION_DIR, dir, "emissions.csv"))
            total_time += data["duration"].values[0]
            total_emissions += data["emissions"].values[0]
            total_kWH += data["energy_consumed"].values[0]

        except FileNotFoundError:
            # print(f"No emissions.csv found in {dir}")
            ...

    minutes, seconds = divmod(total_time, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"Total time: {hours} hours, {minutes} minutes, {int(seconds)} seconds.")
    print(f"Total emissions: {total_emissions:.2f} kgCo2eq.")
    print(f"Total carbon efficiency: {total_emissions / total_kWH:.2f} kgCo2eq / kWH.")
