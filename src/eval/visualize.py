"""src.eval.visualize

Visualization utilities for nowcasting evaluation:
- Attention heatmaps (spatial and temporal)
- Forecast comparison grids
- Animated GIF generation
- Error analysis plots

All functions are matplotlib-based and return figures for flexibility.
Builds on top of src.utils.visualization and src.utils.attention_viz.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def to_numpy(x):
    """Convert tensor to numpy array (handles torch tensors)."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


if TYPE_CHECKING:
    import matplotlib.figure
    import pandas

# Lazy imports for optional dependencies
_plt = None
_sns = None
_imageio = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _get_sns():
    global _sns
    if _sns is None:
        import seaborn as sns
        _sns = sns
    return _sns


def _get_imageio():
    global _imageio
    if _imageio is None:
        import imageio.v2 as imageio
        _imageio = imageio
    return _imageio


# ---------------------------------------------------------------------------
# Attention visualization
# ---------------------------------------------------------------------------


def plot_spatial_attention(
    input_frame: np.ndarray,
    spatial_attn: np.ndarray,
    title: str = "Spatial Attention",
    figsize: Tuple[int, int] = (12, 5),
) -> "matplotlib.figure.Figure":
    """Plot spatial attention importance overlaid with input frame.

    Args:
        input_frame: 2D array (H, W) - single channel input image
        spatial_attn: Attention weights, shape (heads, N, N) or (N, N)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()

    # Average over heads if present
    if spatial_attn.ndim == 3:
        attn_avg = spatial_attn.mean(axis=0)  # (N, N)
    else:
        attn_avg = spatial_attn

    # Sum attention received by each token (column sum)
    importance = attn_avg.sum(axis=0)  # (N,)
    n_side = int(np.sqrt(len(importance)))
    imp_grid = importance.reshape(n_side, n_side)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Input frame
    axes[0].imshow(input_frame, cmap="viridis")
    axes[0].set_title("Input Frame")
    axes[0].axis("off")

    # Attention importance
    im = axes[1].imshow(imp_grid, cmap="hot")
    axes[1].set_title(title)
    plt.colorbar(im, ax=axes[1])
    axes[1].axis("off")

    plt.tight_layout()
    return fig


def plot_temporal_attention(
    temporal_attn: np.ndarray,
    time_labels: Optional[List[str]] = None,
    title: str = "Temporal Attention",
    figsize: Tuple[int, int] = (6, 5),
) -> "matplotlib.figure.Figure":
    """Plot temporal attention matrix as heatmap.

    Args:
        temporal_attn: Attention weights, shape (B*N, heads, T, T) or (heads, T, T) or (T, T)
        time_labels: Optional labels for time steps
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()
    sns = _get_sns()

    # Average to (T, T)
    if temporal_attn.ndim == 4:
        temp_avg = temporal_attn.mean(axis=(0, 1))
    elif temporal_attn.ndim == 3:
        temp_avg = temporal_attn.mean(axis=0)
    else:
        temp_avg = temporal_attn

    T = temp_avg.shape[0]
    if time_labels is None:
        time_labels = [f"t{i}" for i in range(T)]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        temp_avg,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        ax=ax,
        xticklabels=time_labels,
        yticklabels=time_labels,
    )
    ax.set_xlabel("Key (time)")
    ax.set_ylabel("Query (time)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Forecast comparison
# ---------------------------------------------------------------------------


def plot_forecast_comparison(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    channel: int = 0,
    time_step: int = 0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.figure.Figure":
    """Plot side-by-side comparison of multiple model predictions vs ground truth.

    Args:
        ground_truth: Target array, shape (T, C, H, W) or (C, H, W) or (H, W)
        predictions: Dict of {model_name: prediction_array} with same shape
        channel: Channel index to visualize (if C > 1)
        time_step: Time step to visualize (if T > 1)
        vmin, vmax: Color scale limits
        cmap: Colormap name
        figsize: Figure size (auto-computed if None)

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()

    def _extract_frame(arr: np.ndarray) -> np.ndarray:
        """Extract 2D frame from potentially higher-dimensional array."""
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 3:  # (C, H, W)
            return arr[channel]
        elif arr.ndim == 4:  # (T, C, H, W)
            return arr[time_step, channel]
        elif arr.ndim == 5:  # (B, T, C, H, W) - take first sample
            return arr[0, time_step, channel]
        else:
            raise ValueError(f"Cannot extract 2D frame from shape {arr.shape}")

    gt_frame = _extract_frame(ground_truth)
    n_models = len(predictions)
    n_cols = n_models + 1  # +1 for ground truth

    if figsize is None:
        figsize = (4 * n_cols, 4)

    # Auto range
    if vmin is None:
        all_frames = [gt_frame] + [_extract_frame(p) for p in predictions.values()]
        vmin = min(f.min() for f in all_frames)
    if vmax is None:
        all_frames = [gt_frame] + [_extract_frame(p) for p in predictions.values()]
        vmax = max(f.max() for f in all_frames)

    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]

    # Ground truth
    im = axes[0].imshow(gt_frame, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Predictions
    for i, (name, pred) in enumerate(predictions.items(), start=1):
        pred_frame = _extract_frame(pred)
        axes[i].imshow(pred_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i].set_title(name)
        axes[i].axis("off")

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.tight_layout()
    return fig


def plot_error_maps(
    ground_truth: np.ndarray,
    predictions: Dict[str, np.ndarray],
    channel: int = 0,
    time_step: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
) -> "matplotlib.figure.Figure":
    """Plot error maps (prediction - ground truth) for multiple models.

    Args:
        ground_truth: Target array
        predictions: Dict of {model_name: prediction_array}
        channel: Channel index
        time_step: Time step
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()

    def _extract_frame(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return arr
        elif arr.ndim == 3:
            return arr[channel]
        elif arr.ndim == 4:
            return arr[time_step, channel]
        elif arr.ndim == 5:
            return arr[0, time_step, channel]
        else:
            raise ValueError(f"Cannot extract 2D frame from shape {arr.shape}")

    gt_frame = _extract_frame(ground_truth)
    n_models = len(predictions)

    if figsize is None:
        figsize = (4 * n_models, 4)

    # Compute errors
    errors = {}
    for name, pred in predictions.items():
        pred_frame = _extract_frame(pred)
        errors[name] = pred_frame - gt_frame

    # Symmetric colormap range
    max_err = max(np.abs(e).max() for e in errors.values())

    fig, axes = plt.subplots(1, n_models, figsize=figsize)
    if n_models == 1:
        axes = [axes]

    for ax, (name, err) in zip(axes, errors.items()):
        im = ax.imshow(err, cmap="RdBu_r", vmin=-max_err, vmax=max_err)
        ax.set_title(f"{name}\nMAE={np.abs(err).mean():.4f}")
        ax.axis("off")

    plt.colorbar(im, ax=axes, shrink=0.8, label="Error")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def create_forecast_gif(
    frames: Sequence[np.ndarray],
    titles: Sequence[str],
    output_path: Union[str, Path],
    duration: float = 0.5,
    cmap: str = "viridis",
    figsize: Tuple[int, int] = (6, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """Create animated GIF from a sequence of frames.

    Args:
        frames: List of 2D arrays to animate
        titles: List of titles for each frame
        output_path: Where to save the GIF
        duration: Duration of each frame in seconds
        cmap: Colormap
        figsize: Size of each frame
        vmin, vmax: Color scale limits (auto if None)

    Returns:
        Path to saved GIF
    """
    plt = _get_plt()
    imageio = _get_imageio()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if vmin is None:
        vmin = min(f.min() for f in frames)
    if vmax is None:
        vmax = max(f.max() for f in frames)

    gif_frames = []
    for frame, title in zip(frames, titles):
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(frame, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        fig.canvas.draw()
        # Use buffer_rgba for modern matplotlib compatibility
        img = np.asarray(fig.canvas.buffer_rgba())
        img = img[:, :, :3]  # Drop alpha channel
        gif_frames.append(img)
        plt.close(fig)

    imageio.mimsave(str(output_path), gif_frames, duration=duration)
    return output_path


def create_comparison_gif(
    input_seq: np.ndarray,
    predictions: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    output_path: Union[str, Path],
    channel: int = 0,
    duration: float = 0.5,
    t_in: Optional[int] = None,
) -> Path:
    """Create animated GIF showing input → predictions → ground truth.

    Args:
        input_seq: Input sequence, shape (T_in, C, H, W)
        predictions: Dict of {model_name: (T_out, C, H, W)}
        ground_truth: Target sequence, shape (T_out, C, H, W)
        output_path: Where to save the GIF
        channel: Channel to visualize
        duration: Frame duration
        t_in: Number of input frames (inferred if None)

    Returns:
        Path to saved GIF
    """
    if t_in is None:
        t_in = input_seq.shape[0]

    frames = []
    titles = []

    # Input frames
    for t in range(t_in):
        frames.append(input_seq[t, channel])
        titles.append(f"Input t={t}")

    # Ground truth frames
    t_out = ground_truth.shape[0]
    for t in range(t_out):
        frames.append(ground_truth[t, channel])
        titles.append(f"GT t={t_in + t}")

    # Prediction frames (cycle through models)
    for name, pred in predictions.items():
        for t in range(pred.shape[0]):
            frames.append(pred[t, channel])
            titles.append(f"{name} t={t_in + t}")

    return create_forecast_gif(frames, titles, output_path, duration=duration)


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    learning_rates: Optional[List[float]] = None,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 4),
) -> "matplotlib.figure.Figure":
    """Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per step/epoch
        val_losses: Optional validation losses
        learning_rates: Optional learning rates per step
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()

    n_plots = 1 + (val_losses is not None) + (learning_rates is not None)
    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Training loss
    axes[0].plot(train_losses, label="Train", alpha=0.8)
    if val_losses is not None:
        # Interpolate val_losses to match train_losses length for overlay
        val_x = np.linspace(0, len(train_losses) - 1, len(val_losses))
        axes[0].plot(val_x, val_losses, label="Val", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    if learning_rates is not None and len(axes) > 1:
        idx = 1 if val_losses is None else 1
        if idx < len(axes):
            axes[idx].plot(learning_rates, color="green", alpha=0.8)
            axes[idx].set_xlabel("Step")
            axes[idx].set_ylabel("LR")
            axes[idx].set_title("Learning Rate")
            axes[idx].grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_hyperparameter_search(
    results_df: "pandas.DataFrame",
    metric: str = "val_loss",
    param_cols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> "matplotlib.figure.Figure":
    """Plot hyperparameter search results.

    Args:
        results_df: DataFrame with columns for params and metrics
        metric: Metric column to plot
        param_cols: Parameter columns to visualize
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt = _get_plt()

    if param_cols is None:
        # Infer param columns (exclude metric columns)
        metric_cols = {"val_loss", "train_loss", "n_params", "time"}
        param_cols = [c for c in results_df.columns if c not in metric_cols]

    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=figsize)
    if n_params == 1:
        axes = [axes]

    best_idx = results_df[metric].idxmin()
    colors = ["green" if i == best_idx else "steelblue" for i in results_df.index]

    for ax, param in zip(axes, param_cols):
        ax.scatter(results_df[param], results_df[metric], c=colors, s=100, alpha=0.7)
        ax.set_xlabel(param)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Hyperparameter Search ({metric})")
    plt.tight_layout()
    return fig
