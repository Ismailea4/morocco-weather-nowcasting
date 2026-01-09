"""
Visualization script for model comparison.

Creates side-by-side comparison plots of baseline vs ViT predictions.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_npz(path):
    """Load forecast NPZ file."""
    data = np.load(path)
    return data['pred'], data['target']


def plot_channel_comparison(baseline_pred, vit_pred, target, channel_idx=0, 
                           timestep_idx=0, sample_idx=0, output_path="comparison.png"):
    """Create side-by-side comparison plot for a specific channel."""
    
    # Extract the specific sample, timestep, and channel
    baseline_img = baseline_pred[sample_idx, timestep_idx, channel_idx]
    vit_img = vit_pred[sample_idx, timestep_idx, channel_idx]
    target_img = target[sample_idx, timestep_idx, channel_idx]
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot baseline
    im0 = axes[0, 0].imshow(baseline_img, cmap='viridis')
    axes[0, 0].set_title(f'Baseline Prediction\nChannel {channel_idx}, T+{timestep_idx}')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    
    # Plot ViT
    im1 = axes[0, 1].imshow(vit_img, cmap='viridis')
    axes[0, 1].set_title(f'ViT Prediction\nChannel {channel_idx}, T+{timestep_idx}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    # Plot target
    im2 = axes[1, 0].imshow(target_img, cmap='viridis')
    axes[1, 0].set_title(f'Ground Truth\nChannel {channel_idx}, T+{timestep_idx}')
    axes[1, 0].axis('off')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)
    
    # Plot error differences
    baseline_error = np.abs(baseline_img - target_img)
    vit_error = np.abs(vit_img - target_img)
    error_diff = baseline_error - vit_error  # Positive = ViT better
    
    im3 = axes[1, 1].imshow(error_diff, cmap='RdBu', vmin=-np.max(np.abs(error_diff)), 
                            vmax=np.max(np.abs(error_diff)))
    axes[1, 1].set_title(f'Error Difference\n(Blue=ViT better, Red=Baseline better)')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_path}")


def plot_metrics_comparison(baseline_metrics_path, vit_metrics_path, output_path="metrics_comparison.png"):
    """Create bar chart comparing metrics."""
    
    with open(baseline_metrics_path, 'r') as f:
        baseline_metrics = json.load(f)['summary']
    
    with open(vit_metrics_path, 'r') as f:
        vit_metrics = json.load(f)['summary']
    
    metrics = ['rmse', 'mae', 'ssim', 'csi', 'pod', 'far']
    baseline_values = [baseline_metrics[m] for m in metrics]
    vit_values = [vit_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    ax.bar(x + width/2, vit_values, width, label='ViT', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison: Baseline vs ViT')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotations showing which is better
    for i, metric in enumerate(metrics):
        b_val = baseline_values[i]
        v_val = vit_values[i]
        
        # Determine better model (lower is better for rmse, mae, far; higher for others)
        if metric in ['rmse', 'mae', 'far']:
            better = 'ViT' if v_val < b_val else 'Baseline'
        else:
            better = 'ViT' if v_val > b_val else 'Baseline'
        
        y_pos = max(b_val, v_val) + 0.05
        ax.text(i, y_pos, f'✓ {better}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics comparison to {output_path}")


def plot_training_curves(baseline_metrics_path, vit_metrics_path, output_path="training_curves.png"):
    """Plot training and validation loss curves."""
    
    with open(baseline_metrics_path, 'r') as f:
        baseline_data = json.load(f)
    
    with open(vit_metrics_path, 'r') as f:
        vit_data = json.load(f)
    
    baseline_history = baseline_data['history']
    vit_history = vit_data['history']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline curves
    baseline_epochs = [h['epoch'] for h in baseline_history]
    baseline_train = [h['train_loss'] for h in baseline_history]
    baseline_val = [h['val_loss'] for h in baseline_history]
    
    axes[0].plot(baseline_epochs, baseline_train, 'o-', label='Train Loss', alpha=0.7)
    axes[0].plot(baseline_epochs, baseline_val, 's-', label='Val Loss', alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Baseline ConvLSTM Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ViT curves
    vit_epochs = [h['epoch'] for h in vit_history]
    vit_train = [h['train_loss'] for h in vit_history]
    vit_val = [h['val_loss'] for h in vit_history]
    
    axes[1].plot(vit_epochs, vit_train, 'o-', label='Train Loss', alpha=0.7)
    axes[1].plot(vit_epochs, vit_val, 's-', label='Val Loss', alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('ViT Nowcaster Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize model comparison")
    parser.add_argument("--baseline-forecast", default="experiments/baseline_comparison/forecasts_baseline.npz")
    parser.add_argument("--vit-forecast", default="experiments/vit_comparison/forecasts_vit.npz")
    parser.add_argument("--baseline-metrics", default="experiments/baseline_comparison/metrics_baseline.json")
    parser.add_argument("--vit-metrics", default="experiments/vit_comparison/metrics_vit.json")
    parser.add_argument("--output-dir", default="experiments/comparison/visualizations")
    parser.add_argument("--channel", type=int, default=0, help="Channel to visualize")
    parser.add_argument("--timestep", type=int, default=0, help="Timestep to visualize")
    parser.add_argument("--sample", type=int, default=0, help="Sample index to visualize")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading forecasts...")
    baseline_pred, baseline_target = load_npz(args.baseline_forecast)
    vit_pred, vit_target = load_npz(args.vit_forecast)
    
    print(f"Baseline predictions shape: {baseline_pred.shape}")
    print(f"ViT predictions shape: {vit_pred.shape}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Channel comparison
    plot_channel_comparison(
        baseline_pred, vit_pred, baseline_target,
        channel_idx=args.channel,
        timestep_idx=args.timestep,
        sample_idx=args.sample,
        output_path=os.path.join(args.output_dir, f"channel_{args.channel}_t{args.timestep}_comparison.png")
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        args.baseline_metrics, args.vit_metrics,
        output_path=os.path.join(args.output_dir, "metrics_comparison.png")
    )
    
    # Training curves
    plot_training_curves(
        args.baseline_metrics, args.vit_metrics,
        output_path=os.path.join(args.output_dir, "training_curves.png")
    )
    
    print(f"\nAll visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
