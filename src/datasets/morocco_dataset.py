"""
Dataset loader for combined satellite+wind stacks.
Provides sequence sampling for nowcasting tasks.
"""
from pathlib import Path
import glob
import numpy as np
from typing import Tuple, List

class MoroccoWeatherDataset:
    def __init__(self, data_dir: str = "data/combined", sequence_length: int = 6):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.files = sorted(glob.glob(str(self.data_dir / "combined_*.npz")))
        self.sequences = [
            {"input_files": self.files[i:i+sequence_length], "target_file": self.files[i+sequence_length]}
            for i in range(len(self.files) - sequence_length)
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        seq = self.sequences[idx]
        X = [np.load(f)["data"] for f in seq["input_files"]]
        X = np.stack(X, axis=0)
        y = np.load(seq["target_file"]) ["data"]
        return X, y
