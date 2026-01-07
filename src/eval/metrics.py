"""
Evaluation metrics (stub): RMSE, CSI, POD, FAR.
"""
import numpy as np

def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - target) ** 2)))

# TODO: add precipitation thresholding and compute CSI, POD, FAR
