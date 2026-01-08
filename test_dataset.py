import sys
sys.path.append('src')

import torch
from torch.utils.data import DataLoader
from morocco_datasets.morocco_dataset import MoroccoWeatherDataset

# Test the dataset
data_dir = 'dataset'
dataset = MoroccoWeatherDataset(data_dir, sequence_length=6, forecast_horizon=1, split='train')

print(f"Dataset length: {len(dataset)}")

# Test DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for i, (X, y) in enumerate(dataloader):
    print(f"Batch {i}: X shape {X.shape}, y shape {y.shape}")
    if i >= 2:  # Test a few batches
        break

print("DataLoader test passed!")