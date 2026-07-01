import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt


def load_dataset(Dataset):

    """
    Loads the dataset from the specified folder and returns a list of datasets.

    Arguments:
    Dataset -- Name of the dataset folder to load.

    Returns:
    A list of datasets loaded from the specified folder. Each dataset is a dictionary containing cycle data
    """


    data_folder = Path("Battery_life_Dataset/",Dataset)

    pkl_files = list(data_folder.glob("*.pkl"))
    selected_files = pkl_files  # Use all files for now, you can uncomment the next line to select a random sample
    # selected_files = random.sample(pkl_files, 100)
    return selected_files

    datasets = []

    # for pkl_file in selected_files:
    #     with open(pkl_file, "rb") as f:
    #         datasets.append(pickle.load(f))

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