import os
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import csv
from datasets.morocco_dataset import MoroccoWeatherDataset
from models.conv_lstm_baseline import ConvLSTMBaseline

def train_baseline():
    print("Starting training...")
    # Hyperparameters
    config = {
        'data_dir': 'dataset',
        'sequence_length': 3,
        'forecast_horizon': 1,
        'batch_size': 1,  # Small batch due to large images
        'num_epochs': 10,
        'lr': 1e-3,
        'hidden_dim': 16,
        'kernel_size': [3, 3],
        'num_layers': 1,
        'input_channels': 4,  # Adjust to data: 4 channels
        'experiment_dir': 'experiments/baseline_run_001'
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
    model = ConvLSTMBaseline(input_channels=config['input_channels'], hidden_dim=config['hidden_dim'],
                             kernel_size=tuple(config['kernel_size']), num_layers=config['num_layers'])
    model = model.cuda() if torch.cuda.is_available() else model

    # Loss
    criterion = nn.MSELoss()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

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

        # Scheduler step
        scheduler.step(val_loss)

        # Log
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, train_loss, val_loss])

        print(f"Epoch {epoch+1}/{config['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save model
        torch.save(model.state_dict(), os.path.join(config['experiment_dir'], 'model.pt'))
        print(f"Model saved with val_loss: {val_loss:.4f}")

if __name__ == '__main__':
    train_baseline()