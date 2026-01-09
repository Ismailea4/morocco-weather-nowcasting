import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
import json
from glob import glob

class MoroccoWeatherDataset(Dataset):
    def __init__(self, data_dir, sequence_length=6, forecast_horizon=1, split='train'):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.split = split
        
        # Load combined files
        combined_dir = os.path.join(data_dir, 'morocco_processed_dataset', 'combined')
        self.file_list = sorted(glob(os.path.join(combined_dir, '*.npz')))
        
        # Extract timestamps from filenames
        self.timestamps = [os.path.basename(f).replace('combined_', '').replace('.npz', '') for f in self.file_list]
        
        # Temporal split: 70% train, 15% val, 15% test
        n_total = len(self.file_list)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        if split == 'train':
            self.indices = list(range(n_train))
        elif split == 'val':
            self.indices = list(range(n_train, n_train + n_val))
        elif split == 'test':
            self.indices = list(range(n_train + n_val, n_total))
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
        
        # Normalization stats
        stats_file = os.path.join(data_dir, 'normalization_stats.json')
        if split == 'train':
            self.compute_normalization_stats(stats_file)
        else:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.means = np.array(stats['means'])
            self.stds = np.array(stats['stds'])
    
    def compute_normalization_stats(self, stats_file):
        print("Computing normalization stats on train set...")
        all_data = []
        for idx in self.indices:
            data = np.load(self.file_list[idx])['data']  # (4, 557, 521)
            all_data.append(data)
        
        all_data = np.stack(all_data, axis=0)  # (n_train, 4, 557, 521)
        
        self.means = np.mean(all_data, axis=(0, 2, 3))  # (4,)
        self.stds = np.std(all_data, axis=(0, 2, 3))    # (4,)
        
        stats = {'means': self.means.tolist(), 'stds': self.stds.tolist()}
        with open(stats_file, 'w') as f:
            json.dump(stats, f)
        print(f"Stats saved to {stats_file}")
    
    def __len__(self):
        return len(self.indices) - self.sequence_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        # idx is relative to the split indices
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Load sequence
        X_list = []
        for i in range(start_idx, end_idx):
            data = np.load(self.file_list[i])['data']  # (4, 557, 521)
            X_list.append(data)
        
        X = np.stack(X_list, axis=0)  # (sequence_length, 4, 557, 521)
        
        # Load target
        target_idx = end_idx + self.forecast_horizon - 1
        y = np.load(self.file_list[target_idx])['data']  # (4, 557, 521)
        y = y[np.newaxis, ...]  # (1, 4, 557, 521) for forecast_horizon=1
        
        # Normalize
        X = (X - self.means[None, :, None, None]) / self.stds[None, :, None, None]
        y = (y - self.means[None, :, None, None]) / self.stds[None, :, None, None]
        
        # Convert to torch tensors
        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        
        return X, y