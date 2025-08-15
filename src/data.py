from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesWindowDataset(Dataset):
    '''
    Sequence-to-one dataset that creates sliding windows from a CSV.
    Assumes regression with scalar or vector target.
    '''
    def __init__(
        self,
        csv_path: str,
        features: List[str],
        target: str,
        seq_len: int = 50,
        stride: int = 1,
        scaler: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,  # {"mean": ..., "std": ...}
        output_dim: int = 1,
    ):
        df = pd.read_csv(csv_path)
        self.features = features
        self.target = target
        X = df[features].values.astype(np.float32)
        y = df[[target]].values.astype(np.float32) if output_dim == 1 else df[target].values.astype(np.float32)

        # Scale features if scaler provided
        if scaler is not None:
            mean, std = scaler["mean"], scaler["std"]
            std = np.where(std == 0, 1.0, std)
            X = (X - mean) / std

        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.stride = stride
        self.output_dim = output_dim

        # Precompute valid start indices
        self.idxs = list(range(0, len(self.X) - self.seq_len, self.stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        s = self.idxs[i]
        e = s + self.seq_len
        x = self.X[s:e]                          # [seq_len, feat_dim]
        y = self.y[e] if e < len(self.y) else self.y[-1]  # predict next point
        return torch.from_numpy(x), torch.from_numpy(y)

def compute_standard_scaler(csv_path: str, features: List[str]):
    df = pd.read_csv(csv_path)
    X = df[features].values.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": mean, "std": std}
