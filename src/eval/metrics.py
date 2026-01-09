"""
Comprehensive evaluation metrics for weather nowcasting.

Metrics:
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error
- SSIM: Structural Similarity Index
- CSI: Critical Success Index (for event detection)
- POD: Probability of Detection
- FAR: False Alarm Rate
"""

import numpy as np
from typing import Dict, Any
from skimage.metrics import structural_similarity as ssim


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def ssim_score(pred: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    """Structural Similarity Index.
    
    Args:
        pred: Predicted array of shape (B, T, C, H, W) or similar
        target: Target array of same shape
        data_range: Range of data (default 1.0 for normalized data)
    
    Returns:
        SSIM score
    """
    # Flatten spatial dimensions for SSIM calculation
    pred_flat = pred.reshape(-1, *pred.shape[-2:])  # (B*T*C, H, W)
    target_flat = target.reshape(-1, *target.shape[-2:])
    
    ssim_scores = []
    for p, t in zip(pred_flat, target_flat):
        score = ssim(t, p, data_range=data_range)
        ssim_scores.append(score)
    
    return float(np.mean(ssim_scores))


def csi_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Critical Success Index.
    
    CSI = Hits / (Hits + False Alarms + Misses)
    
    Measures accuracy of detecting events above threshold.
    """
    pred_binary = (pred >= threshold).astype(int)
    target_binary = (target >= threshold).astype(int)
    
    hits = np.sum(pred_binary * target_binary)
    false_alarms = np.sum(pred_binary * (1 - target_binary))
    misses = np.sum((1 - pred_binary) * target_binary)
    
    denominator = hits + false_alarms + misses
    if denominator == 0:
        return 0.0
    
    return float(hits / denominator)


def pod_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Probability of Detection.
    
    POD = Hits / (Hits + Misses)
    
    Sensitivity - what fraction of actual events were detected.
    """
    pred_binary = (pred >= threshold).astype(int)
    target_binary = (target >= threshold).astype(int)
    
    hits = np.sum(pred_binary * target_binary)
    misses = np.sum((1 - pred_binary) * target_binary)
    
    denominator = hits + misses
    if denominator == 0:
        return 0.0
    
    return float(hits / denominator)


def far_score(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """False Alarm Rate.
    
    FAR = False Alarms / (Hits + False Alarms)
    
    Fraction of predicted events that didn't actually occur.
    Lower is better.
    """
    pred_binary = (pred >= threshold).astype(int)
    target_binary = (target >= threshold).astype(int)
    
    hits = np.sum(pred_binary * target_binary)
    false_alarms = np.sum(pred_binary * (1 - target_binary))
    
    denominator = hits + false_alarms
    if denominator == 0:
        return 0.0
    
    return float(false_alarms / denominator)


def per_channel_mae(pred: np.ndarray, target: np.ndarray, channel_axis: int = -3) -> np.ndarray:
    """Compute MAE per channel.
    
    Args:
        pred: Array of shape (..., C, H, W) or similar
        target: Same shape as pred
        channel_axis: Which axis is the channel dimension
    
    Returns:
        Array of shape (C,) with MAE for each channel
    """
    # Normalize channel_axis for negative indices
    ndim = pred.ndim
    if channel_axis < 0:
        channel_axis = ndim + channel_axis
    
    # Get number of channels
    num_channels = pred.shape[channel_axis]
    
    # Compute MAE per channel by reducing over all axes except channel
    mae_per_channel = []
    for c in range(num_channels):
        # Index along channel axis
        if channel_axis == ndim - 3:
            p_c = pred[..., c, :, :]
            t_c = target[..., c, :, :]
        elif channel_axis == ndim - 2:
            p_c = pred[..., c, :]
            t_c = target[..., c, :]
        elif channel_axis == ndim - 1:
            p_c = pred[..., c]
            t_c = target[..., c]
        else:
            # General case: use take
            p_c = np.take(pred, c, axis=channel_axis)
            t_c = np.take(target, c, axis=channel_axis)
        
        mae_per_channel.append(np.mean(np.abs(p_c - t_c)))
    
    return np.array(mae_per_channel, dtype=np.float32)


def per_channel_rmse(pred: np.ndarray, target: np.ndarray, channel_axis: int = -3) -> np.ndarray:
    """Compute RMSE per channel.
    
    Args:
        pred: Array of shape (..., C, H, W) or similar
        target: Same shape as pred
        channel_axis: Which axis is the channel dimension
    
    Returns:
        Array of shape (C,) with RMSE for each channel
    """
    # Normalize channel_axis for negative indices
    ndim = pred.ndim
    if channel_axis < 0:
        channel_axis = ndim + channel_axis
    
    # Get number of channels
    num_channels = pred.shape[channel_axis]
    
    # Compute RMSE per channel
    rmse_per_channel = []
    for c in range(num_channels):
        # Index along channel axis
        if channel_axis == ndim - 3:
            p_c = pred[..., c, :, :]
            t_c = target[..., c, :, :]
        elif channel_axis == ndim - 2:
            p_c = pred[..., c, :]
            t_c = target[..., c, :]
        elif channel_axis == ndim - 1:
            p_c = pred[..., c]
            t_c = target[..., c]
        else:
            # General case
            p_c = np.take(pred, c, axis=channel_axis)
            t_c = np.take(target, c, axis=channel_axis)
        
        rmse_per_channel.append(np.sqrt(np.mean((p_c - t_c) ** 2)))
    
    return np.array(rmse_per_channel, dtype=np.float32)


def summary_table(pred: np.ndarray, target: np.ndarray, 
                 threshold: float = 0.5, data_range: float = 1.0) -> Dict[str, float]:
    """Compute all summary metrics.
    
    Args:
        pred: Prediction array
        target: Target array
        threshold: Threshold for event detection metrics
        data_range: Data range for SSIM
    
    Returns:
        Dictionary with keys: rmse, mae, ssim, csi, pod, far
    """
    return {
        'rmse': rmse(pred, target),
        'mae': mae(pred, target),
        'ssim': ssim_score(pred, target, data_range=data_range),
        'csi': csi_score(pred, target, threshold=threshold),
        'pod': pod_score(pred, target, threshold=threshold),
        'far': far_score(pred, target, threshold=threshold),
    }
