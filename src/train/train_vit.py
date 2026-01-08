import os
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import csv
from morocco_datasets.morocco_dataset import MoroccoWeatherDataset
from models.vit_nowcasting import ViTNowcasting

def train_vit():
    print("Starting ViT training...")
    # Hyperparameters
    config = {
        'data_dir': 'dataset',
        'sequence_length': 3,
        'forecast_horizon': 1,
        'batch_size': 1,  # Small batch due to large images
        'num_epochs': 5,  # Fewer epochs for demo
        'lr': 1e-4,
        'embed_dim': 48,
        'depths': [1, 1, 1, 1],
        'num_heads': [3, 6, 12, 12],
        'window_size': 7,
        'input_channels': 4,
        'experiment_dir': 'experiments/vit_run_001'
    }

    os.makedirs(config['experiment_dir'], exist_ok=True)

    # Save config
    with open(os.path.join(config['experiment_dir'], 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    # Datasets
    train_dataset = MoroccoWeatherDataset(config['data_dir'], config['sequence_length'], config['forecast_horizon'], 'train')
    val_dataset = MoroccoWeatherDataset(config['data_dir'], config['sequence_length'], config['forecast_horizon'], 'val')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # Model
    model = ViTNowcasting(input_channels=config['input_channels'], embed_dim=config['embed_dim'],
                          depths=config['depths'], num_heads=config['num_heads'],
                          window_size=config['window_size'], seq_length=config['sequence_length'],
                          forecast_horizon=config['forecast_horizon'])
    model = model.cuda() if torch.cuda.is_available() else model

    # Loss
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])

    # Metrics
    metrics_file = os.path.join(config['experiment_dir'], 'metrics.csv')
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.cuda() if torch.cuda.is_available() else X
            y = y.cuda() if torch.cuda.is_available() else y

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.cuda() if torch.cuda.is_available() else X
                y = y.cuda() if torch.cuda.is_available() else y
                output = model(X)
                loss = criterion(output, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Log
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss])

        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(config['experiment_dir'], 'model.pt'))
        print(f"Model saved with val_loss: {val_loss:.4f}")

if __name__ == '__main__':
    train_vit()