from sklearn import datasets
import pandas as pd
import numpy as np  
from scipy.signal import find_peaks
from dtaidistance import dtw


def feature_engineering_1(datasets):
    """
    Extracts features from the provided datasets and returns a DataFrame containing the extracted features.

    Arguments:
    datasets -- List of datasets, where each dataset is a dictionary containing cycle data.

    Returns:
    A pandas DataFrame containing the extracted features for each cycle in the datasets.    


    The function calculates the following features for each cycle:
    - SOH (State of Health): Calculated as the ratio of the maximum discharge capacity to the initial maximum discharge capacity (Q0).
    - I_mean: Mean of the current values during the cycle.
    - I_std: Standard deviation of the current values during the cycle.
    - charge_duration: Total duration of charging (time when current is positive).
    - discharge_duration: Total duration of discharging (time when current is negative).
    - V_mean: Mean of the voltage values during the cycle.
    - V_std: Standard deviation of the voltage values during the cycle. 

    The extracted features are stored in a pandas DataFrame, where each row corresponds to a cycle and contains the following columns:
    - cell_id: Identifier of the battery cell.
    - cycle_number: Cycle number of the battery cell.   

    The function uses the first cycle of each dataset to determine the initial maximum discharge capacity (Q0) for calculating SOH.
    """
    features = []
    for data in datasets:

        cycles = data["cycle_data"]

        Q0 = np.max(cycles[1]["discharge_capacity_in_Ah"])

        for cycle in cycles:

            I = np.array(cycle["current_in_A"], dtype=np.float32)
            V = np.array(cycle["voltage_in_V"], dtype=np.float32)
            t = np.array(cycle["time_in_s"], dtype=np.float32)

            # SOH
            Q_discharge = np.max(cycle["discharge_capacity_in_Ah"])
            soh = Q_discharge / Q0

            # Time differences for charge and discharge duration calculations
            dt = np.diff(t, prepend=t[0])

            charge_duration = np.sum(dt[I > 0])
            discharge_duration = np.sum(dt[I < 0])

            features.append({
                "cell_id": data["cell_id"],
                "cycle_number": cycle["cycle_number"],

                "SOH": soh,

                "I_mean": np.mean(I),
                "I_std": np.std(I),

                "charge_duration": charge_duration,
                "discharge_duration": discharge_duration,

                "V_mean": np.mean(V),
                "V_std": np.std(V)
            })

    feature_df = pd.DataFrame(features)
    return feature_df




def feature_engineering_2(datasets):
    """
    Extracts voltage shape features from the provided datasets and returns a DataFrame containing the extracted features.
    
    Arguments:
    datasets -- List of datasets, where each dataset is a dictionary containing cycle data.
    
    Returns:
    A pandas DataFrame containing the extracted voltage shape features for each cycle in the datasets.
    
    The function calculates the following voltage shape features for each cycle:
    - V_range: The range of voltage values (max - min).
    - V_slope_mean: The mean of the absolute values of the voltage slope (first derivative).
    - V_curvature: The mean of the absolute values of the voltage curvature (second derivative).
    - V_n_peaks: The number of peaks in the voltage signal.
    - RMSE_V: Root Mean Square Error between the reference voltage and the current cycle voltage.
    - AreaDiff_V: Mean absolute difference between the reference voltage and the current cycle voltage.
    - Corr_V: Correlation coefficient between the reference voltage and the current cycle voltage.
    - MaxDev_V: Maximum absolute deviation between the reference voltage and the current cycle voltage.
    - SlopeRMSE_V: Root Mean Square Error between the slopes of the reference voltage and the current cycle voltage.
    
    The extracted features are stored in a pandas DataFrame, where each row corresponds to a cycle and contains the following columns:
    - cell_id: Identifier of the battery cell.
    - cycle_number: Cycle number of the battery cell.
            
    
    The function uses the first cycle of each dataset as the reference voltage for comparison with subsequent cycles.
    """
    v_shape_features = []
    for data in datasets:
        ref_v = np.array(
            data["cycle_data"][1]["voltage_in_V"],
            dtype=np.float32
        )
        for cycle in data["cycle_data"]:

            v = np.array(cycle["voltage_in_V"], dtype=np.float32)

            dv = np.diff(v)
            d2v = np.diff(dv)

            peaks, _ = find_peaks(v)

            # Ensure both signals have the same length
            min_len = min(len(ref_v), len(v))

            ref = ref_v[:min_len]
            x = v[:min_len]

            # simple shape features
            rmse_v = np.sqrt(np.mean((ref - x) ** 2))

            area_diff_v = np.mean(np.abs(ref - x))

            corr_v = np.corrcoef(ref, x)[0, 1]

            max_dev_v = np.max(np.abs(ref - x))

            # Compare derivatives of the reference and current cycle voltage
            dref = np.diff(ref)
            dx = np.diff(x)

            min_len_diff = min(len(dref), len(dx))

            slope_rmse_v = np.sqrt(
                np.mean(
                    (dref[:min_len_diff] - dx[:min_len_diff]) ** 2
                )
            )

            v_shape_features.append({
                "cell_id": data["cell_id"],
                "cycle_number": cycle["cycle_number"],

                # bisherige Features
                "V_range": np.max(v) - np.min(v),
                "V_slope_mean": np.mean(np.abs(dv)),
                "V_curvature": np.mean(np.abs(d2v)),
                "V_n_peaks": len(peaks),

                # neue DTW-Alternativen
                "RMSE_V": rmse_v,
                "AreaDiff_V": area_diff_v,
                "Corr_V": corr_v,
                "MaxDev_V": max_dev_v,
                "SlopeRMSE_V": slope_rmse_v
            })
    v_shape_df = pd.DataFrame(v_shape_features)
    return v_shape_df




def feature_engineering_dtw_current(datasets):
    """
    Computes DTW-based similarity features for current curves of battery cycles.

    Arguments:
    datasets -- List of datasets, where each dataset is a dictionary containing cycle data.

    Returns:
    A pandas DataFrame with DTW distances between each cycle and the reference cycle.

    The function calculates:
    - DTW_I: Dynamic Time Warping distance between normalized current signals of
             each cycle and the reference cycle (cycle 1).
    """

    dtw_features = []

    for data in datasets:

        cycles = data["cycle_data"]

        # reference cycle
        ref = np.array(cycles[1]["current_in_A"], dtype=np.float32)
        ref = (ref - np.mean(ref)) / (np.std(ref) + 1e-8)

        for cycle in cycles:

            x = np.array(cycle["current_in_A"], dtype=np.float32)
            x = (x - np.mean(x)) / (np.std(x) + 1e-8)

            dist = dtw.distance(ref, x)

            dtw_features.append({
                "cell_id": data["cell_id"],
                "cycle_number": cycle["cycle_number"],
                "DTW_I": dist
            })

    return pd.DataFrame(dtw_features)



def feature_engineering_dtw_voltage(datasets):
    """
    Computes DTW-based similarity features for voltage curves of battery cycles.

    Arguments:
    datasets -- List of datasets, where each dataset is a dictionary containing cycle data.

    Returns:
    A pandas DataFrame with DTW distances between each cycle voltage curve and the reference cycle.

    The function calculates:
    - DTW_V: Dynamic Time Warping distance between normalized voltage signals of
             each cycle and the reference cycle (cycle 1).
    """

    dtw_v_features = []

    for data in datasets:

        cycles = data["cycle_data"]

        # reference cycle (as in your code)
        ref_v = np.array(cycles[1]["voltage_in_V"], dtype=np.float32)
        ref_v = (ref_v - np.mean(ref_v)) / (np.std(ref_v) + 1e-8)

        for cycle in cycles:

            v = np.array(cycle["voltage_in_V"], dtype=np.float32)
            v = (v - np.mean(v)) / (np.std(v) + 1e-8)

            dist_v = dtw.distance(ref_v, v)

            dtw_v_features.append({
                "cell_id": data["cell_id"],
                "cycle_number": cycle["cycle_number"],
                "DTW_V": dist_v
            })

    return pd.DataFrame(dtw_v_features)


