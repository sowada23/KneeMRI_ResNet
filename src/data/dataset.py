from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class KneeDataset(Dataset):
    def __init__(self, data_root, tfm=None):
        self.tfm = tfm
        self.data_root = Path(data_root)
        self.samples = []

        for npy_path in self.data_root.rglob("*.npy"):
            class_name = npy_path.parent.parent.name
            self.samples.append((npy_path, int(class_name)))   # assumes class folders are "0" and "1"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        p = Path(path)
        pid = f"{p.parent.parent.name}/{p.parent.name}"

        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got {arr.shape} for {path}")
        
        x = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W]

        if self.tfm is not None:
            x = self.tfm(x)
            
        y = torch.tensor(y, dtype=torch.float32)
        return x, y, pid


def collate_with_pid(batch):
    xs, ys, pids = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), list(pids)