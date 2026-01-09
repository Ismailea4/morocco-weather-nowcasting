"""
Unified training script for baseline ConvLSTM and ViT models.

Trains both models on the same dataset and generates comparison artifacts.

Usage:
    python src/train/train_both_models.py --baseline-epochs 20 --vit-epochs 10
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.conv_lstm_baseline import ConvLSTMBaseline
from src.models.vit_nowcasting import ViTNowcaster, ViTNowcasterConfig
from src.eval.artifacts import save_forecast_npz
from src.eval.metrics import summary_table, per_channel_mae, per_channel_rmse


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_dir="data/combined", batch_size=4):
    """Load train, val, test datasets.
    
    Handles the actual data structure where combined NPZ files are directly in data_dir.
    """
    # Create a simple wrapper dataset that works with your actual data structure
    class SimpleWeatherDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, sequence_length=6, forecast_horizon=1, means=None, stds=None):
            self.file_list = file_list
            self.sequence_length = sequence_length
            self.forecast_horizon = forecast_horizon
            self.means = means
            self.stds = stds
            
            # Compute normalization stats if not provided
            if means is None or stds is None:
                print("Computing normalization stats...")
                all_data = []
                for f in file_list:
                    data = np.load(f)['data']
                    all_data.append(data)
                all_data = np.stack(all_data, axis=0)
                self.means = np.mean(all_data, axis=(0, 2, 3))
                self.stds = np.std(all_data, axis=(0, 2, 3))
                self.stds[self.stds == 0] = 1.0  # Avoid division by zero
        
        def __len__(self):
            return max(0, len(self.file_list) - self.sequence_length - self.forecast_horizon + 1)
        
        def __getitem__(self, idx):
            # Load sequence
            X_list = []
            for i in range(idx, idx + self.sequence_length):
                if i >= len(self.file_list):
                    raise IndexError(f"Index {i} out of range")
                data = np.load(self.file_list[i])['data']
                X_list.append(data)
            
            X = np.stack(X_list, axis=0)
            
            # Load target
            target_idx = idx + self.sequence_length + self.forecast_horizon - 1
            if target_idx >= len(self.file_list):
                raise IndexError(f"Target index {target_idx} out of range")
            y = np.load(self.file_list[target_idx])['data']
            y = y[np.newaxis, ...]
            
            # Normalize
            X = (X - self.means[None, :, None, None]) / self.stds[None, :, None, None]
            y = (y - self.means[None, :, None, None]) / self.stds[None, :, None, None]
            
            X = torch.from_numpy(X).float()
            y = torch.from_numpy(y).float()
            
            return X, y
    
    # Get combined files from the actual data directory
    combined_files = sorted(list(Path(data_dir).glob("combined_*.npz")))
    
    if not combined_files:
        raise FileNotFoundError(f"No combined_*.npz files found in {data_dir}")
    
    print(f"Found {len(combined_files)} combined files in {data_dir}")
    
    # Compute normalization stats from all files
    all_data = []
    for f in combined_files:
        data = np.load(f)['data']
        all_data.append(data)
    all_data = np.stack(all_data, axis=0)
    means = np.mean(all_data, axis=(0, 2, 3))
    stds = np.std(all_data, axis=(0, 2, 3))
    stds[stds == 0] = 1.0
    
    # Temporal split: 70% train, 15% val, 15% test
    n_total = len(combined_files)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    
    train_files = combined_files[:n_train]
    val_files = combined_files[n_train:n_train + n_val]
    test_files = combined_files[n_train + n_val:]
    
    train_dataset = SimpleWeatherDataset(train_files, means=means, stds=stds)
    val_dataset = SimpleWeatherDataset(val_files, means=means, stds=stds)
    test_dataset = SimpleWeatherDataset(test_files, means=means, stds=stds)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def train_baseline(train_loader, val_loader, test_loader, epochs=20, lr=1e-3, exp_dir="experiments/baseline_comparison"):
    """Train ConvLSTM baseline."""
    print("\n" + "="*60)
    print("TRAINING BASELINE CONVLSTM")
    print("="*60)
    
    os.makedirs(exp_dir, exist_ok=True)
    device = get_device()
    
    # Get input/output shapes from first batch
    X, y = next(iter(train_loader))
    seq_len, channels, h, w = X.shape[1:]
    t_out = y.shape[1]
    
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")
    print(f"seq_len={seq_len}, channels={channels}, h={h}, w={w}, t_out={t_out}")
    
    model = ConvLSTMBaseline(
        input_channels=channels,
        hidden_dim=64,
        kernel_size=(3, 3),
        num_layers=3
    ).to(device)
    
    print(f"Model created. Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_loss = float('inf')
    metrics_history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        try:
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                print(f"  Batch {batch_idx}: X {X_batch.shape}, y {y_batch.shape}")
                
                optimizer.zero_grad()
                pred = model(X_batch)
                print(f"  Pred shape: {pred.shape}")
                loss = criterion(pred, y_batch)
                print(f"  Loss: {loss.item():.4f}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                print(f"  Batch {batch_idx} complete")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step()
        
        # Log
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        metrics_history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "baseline_best.pt"))
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, "baseline_best.pt")))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds, axis=0)  # (N, t_out, C, H, W)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = summary_table(preds, targets, threshold=0.5)
    pcm = per_channel_mae(preds, targets)
    pcr = per_channel_rmse(preds, targets)
    
    print(f"\nTest Metrics (Baseline):")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  CSI: {metrics['csi']:.4f}")
    print(f"  POD: {metrics['pod']:.4f}")
    print(f"  FAR: {metrics['far']:.4f}")
    
    # Save artifacts
    forecast_path = os.path.join(exp_dir, "forecasts_baseline.npz")
    meta = {"model": "baseline", "epochs": epochs}
    save_forecast_npz(forecast_path, preds, targets, meta)
    
    # Save metrics
    with open(os.path.join(exp_dir, "metrics_baseline.json"), 'w') as f:
        json.dump({
            "summary": metrics,
            "per_channel_mae": pcm.tolist(),
            "per_channel_rmse": pcr.tolist(),
            "history": metrics_history
        }, f, indent=2)
    
    return preds, targets, metrics, forecast_path


def train_vit(train_loader, val_loader, test_loader, epochs=10, lr=1e-3, exp_dir="experiments/vit_comparison"):
    """Train ViT model."""
    print("\n" + "="*60)
    print("TRAINING VIT NOWCASTER")
    print("="*60)
    
    os.makedirs(exp_dir, exist_ok=True)
    device = get_device()
    
    # Get shapes from batch
    X, y = next(iter(train_loader))
    _, channels, h, w = X.shape[1:]
    
    # Create config
    config = ViTNowcasterConfig(
        image_size=h,
        patch_size=16,
        in_channels=channels,
        out_channels=channels,
        t_out=y.shape[1],
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        temporal_fusion='attention',
        temporal_heads=4
    )
    
    model = ViTNowcaster(config).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Warmup + cosine annealing
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)  # 1 epoch warmup
    
    def warmup_cosine_scheduler(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine_scheduler)
    
    best_val_loss = float('inf')
    metrics_history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Reshape for ViT: (B, T, C, H, W)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Log
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        metrics_history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "vit_best.pt"))
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(os.path.join(exp_dir, "vit_best.pt")))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    metrics = summary_table(preds, targets, threshold=0.5)
    pcm = per_channel_mae(preds, targets)
    pcr = per_channel_rmse(preds, targets)
    
    print(f"\nTest Metrics (ViT):")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  CSI: {metrics['csi']:.4f}")
    print(f"  POD: {metrics['pod']:.4f}")
    print(f"  FAR: {metrics['far']:.4f}")
    
    # Save artifacts
    forecast_path = os.path.join(exp_dir, "forecasts_vit.npz")
    meta = {"model": "vit", "epochs": epochs}
    save_forecast_npz(forecast_path, preds, targets, meta)
    
    # Save metrics
    with open(os.path.join(exp_dir, "metrics_vit.json"), 'w') as f:
        json.dump({
            "summary": metrics,
            "per_channel_mae": pcm.tolist(),
            "per_channel_rmse": pcr.tolist(),
            "history": metrics_history
        }, f, indent=2)
    
    return preds, targets, metrics, forecast_path


def create_comparison_report(baseline_metrics, vit_metrics, baseline_preds, vit_preds, 
                            baseline_targets, vit_targets, output_dir="experiments/comparison"):
    """Create detailed comparison report."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPARISON REPORT")
    print("="*60)
    
    md_lines = [
        "# Model Comparison Report\n",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Summary Metrics Comparison\n\n",
        "| Model | RMSE ↓ | MAE ↓ | SSIM ↑ | CSI ↑ | POD ↑ | FAR ↓ |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]
    
    # Baseline row
    md_lines.append(
        f"| Baseline | {baseline_metrics['rmse']:.4f} | {baseline_metrics['mae']:.4f} | "
        f"{baseline_metrics['ssim']:.4f} | {baseline_metrics['csi']:.4f} | "
        f"{baseline_metrics['pod']:.4f} | {baseline_metrics['far']:.4f} |\n"
    )
    
    # ViT row
    md_lines.append(
        f"| ViT | {vit_metrics['rmse']:.4f} | {vit_metrics['mae']:.4f} | "
        f"{vit_metrics['ssim']:.4f} | {vit_metrics['csi']:.4f} | "
        f"{vit_metrics['pod']:.4f} | {vit_metrics['far']:.4f} |\n"
    )
    
    # Differences
    md_lines.append("\n## Differences (ViT - Baseline)\n\n")
    md_lines.append("| Metric | Difference | Winner |\n")
    md_lines.append("|---|---:|---|\n")
    
    metrics_to_compare = ['rmse', 'mae', 'ssim', 'csi', 'pod', 'far']
    
    for metric in metrics_to_compare:
        diff = vit_metrics[metric] - baseline_metrics[metric]
        
        # Determine winner based on metric direction
        if metric in ['ssim', 'csi', 'pod']:  # Higher is better
            winner = "ViT" if diff > 0 else "Baseline"
        else:  # Lower is better
            winner = "Baseline" if diff > 0 else "ViT"
        
        md_lines.append(f"| {metric.upper()} | {diff:+.4f} | {winner} |\n")
    
    # Per-channel comparison
    pcm_baseline = per_channel_mae(baseline_preds, baseline_targets)
    pcm_vit = per_channel_mae(vit_preds, vit_targets)
    pcr_baseline = per_channel_rmse(baseline_preds, baseline_targets)
    pcr_vit = per_channel_rmse(vit_preds, vit_targets)
    
    md_lines.append("\n## Per-Channel Error Analysis\n\n")
    md_lines.append("| Channel | Baseline MAE | ViT MAE | Baseline RMSE | ViT RMSE |\n")
    md_lines.append("|---|---:|---:|---:|---:|\n")
    
    for ch in range(len(pcm_baseline)):
        md_lines.append(
            f"| {ch} | {pcm_baseline[ch]:.4f} | {pcm_vit[ch]:.4f} | "
            f"{pcr_baseline[ch]:.4f} | {pcr_vit[ch]:.4f} |\n"
        )
    
    # Key insights
    md_lines.append("\n## Key Insights\n\n")
    
    rmse_winner = "ViT" if vit_metrics['rmse'] < baseline_metrics['rmse'] else "Baseline"
    detection_winner = "ViT" if vit_metrics['pod'] > baseline_metrics['pod'] else "Baseline"
    
    md_lines.append(f"- **Continuous Metrics**: {rmse_winner} performs better on pixel-level regression (RMSE, MAE, SSIM)\n")
    md_lines.append(f"- **Event Detection**: {detection_winner} performs better on detecting weather events (CSI, POD)\n")
    md_lines.append(f"- **Channel Analysis**: Review per-channel errors to identify if models struggle with specific data types\n")
    
    report_text = "".join(md_lines)
    report_path = os.path.join(output_dir, "comparison_report.md")
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"Report saved to: {report_path}")
    
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/combined", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--baseline-epochs", type=int, default=20, help="Epochs for baseline")
    parser.add_argument("--vit-epochs", type=int, default=10, help="Epochs for ViT")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--baseline-only", action="store_true", help="Train only baseline")
    parser.add_argument("--vit-only", action="store_true", help="Train only ViT")
    args = parser.parse_args()
    
    # Load data
    train_loader, val_loader, test_loader = load_data(args.data_dir, args.batch_size)
    
    baseline_metrics = None
    vit_metrics = None
    baseline_preds = None
    vit_preds = None
    baseline_targets = None
    vit_targets = None
    
    # Train models
    if not args.vit_only:
        baseline_preds, baseline_targets, baseline_metrics, baseline_forecast = train_baseline(
            train_loader, val_loader, test_loader, 
            epochs=args.baseline_epochs, 
            lr=args.lr
        )
    
    if not args.baseline_only:
        vit_preds, vit_targets, vit_metrics, vit_forecast = train_vit(
            train_loader, val_loader, test_loader, 
            epochs=args.vit_epochs, 
            lr=args.lr
        )
    
    # Create comparison report if both trained
    if baseline_metrics and vit_metrics:
        create_comparison_report(
            baseline_metrics, vit_metrics,
            baseline_preds, vit_preds,
            baseline_targets, vit_targets
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
