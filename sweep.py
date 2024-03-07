"""
Create a weights & biases sweep file based on the .yaml config file specified as an argument to the script.
Then sets the sweep ID as a environment variable so that it can be accessed easily.
"""

# STD
import os
import yaml
import sys

# EXT
import wandb

# PROJECT
from src.constants import PROJECT_NAME

try:
    from secret import WANDB_API_KEY, WANDB_USER_NAME

    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

except ModuleNotFoundError:
    WANDB_USER_NAME = os.environ["WANDB_API_KEY"]


if __name__ == "__main__":
    wandb.init(PROJECT_NAME)

    # Get path to sweep .yaml
    config_yaml = sys.argv[1]
    num_runs = int(sys.argv[2])

    with open(config_yaml) as file:
        config_dict = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(config_dict, project=PROJECT_NAME)
    wandb.agent(sweep_id, count=num_runs)
