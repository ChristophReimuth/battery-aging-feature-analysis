import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt


def load_dataset(pathstring, n_files="all", seed=42):

    """
    Loads the dataset from the specified folder and returns a list of datasets.

    Arguments:
    pathstring -- Path to the dataset folder to load.
    n_files -- Number of files to load (default is "all" to load all files).
    seed -- Random seed for reproducibility (default is 42).

    Returns:
    A list of datasets loaded from the specified folder. Each dataset is a dictionary containing cycle data
    """


    data_folder = Path(pathstring)
    pkl_files = list(data_folder.glob("*.pkl"))

    if n_files == "all":
        selected_files = pkl_files
    else:
        if n_files > len(pkl_files):
            raise ValueError(
                f"Requested {n_files} files but only {len(pkl_files)} are available."
            )

        random.seed(seed)
        selected_files = random.sample(pkl_files, n_files)

    datasets = []

    for pkl_file in selected_files:
        with open(pkl_file, "rb") as f:
            datasets.append(pickle.load(f))


    return datasets



# Selection of which cycles to keep in the dataset


def reduce_cycles(cycle_data, step=20, keep_cycles=[5, 10, 15, 30]):

    """ 
    Reduces the number of cycles in the dataset based on the specified step size.
    
    Arguments:
    cycle_data -- List of dictionaries containing cycle data.
    step -- Step size for selecting cycles (default is 20). 
    keep_cycles -- List of specific cycle numbers to keep (default is [5, 10, 15, 30]).

    Returns:
    A filtered list of cycle data dictionaries, keeping cycles based on the specified step size.

    The function retains cycles with cycle numbers less than or equal to 5, as well 
    as cycles within keep_cycles. Additionally, it keeps cycles whose cycle numbers are multiples of the specified step size.

    """
    return [
        cycle
        for cycle in cycle_data
        if cycle["cycle_number"] <= 5
        or cycle["cycle_number"] in keep_cycles
        or cycle["cycle_number"] % step == 0
    ]