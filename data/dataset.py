"""Dataset wrappers for pre-tokenized TinyStories chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def cfg_get(config: Any, section: str, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(section, {}).get(key, default)
    section_obj = getattr(config, section, None)
    return getattr(section_obj, key, default) if section_obj is not None else default


class TinyStoriesDataset(Dataset):
    """Memory-mapped fixed-length language modeling dataset."""

    def __init__(self, bin_path: str, seq_len: int = 256):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Missing token cache: {self.bin_path}")
        self.seq_len = seq_len
        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.n_sequences = (len(self.tokens) - 1) // self.seq_len

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = np.asarray(self.tokens[start:end], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


def create_dataloaders(config: Any) -> tuple[DataLoader, DataLoader]:
    seq_len = int(cfg_get(config, "data", "chunk_size", 256))
    cache_dir = Path(cfg_get(config, "data", "tensor_cache_dir", "data/cache"))
    batch_size = int(cfg_get(config, "training", "batch_size", 32))
    num_workers = int(cfg_get(config, "data", "num_workers", 4))

    train_dataset = TinyStoriesDataset(str(cache_dir / "train.bin"), seq_len=seq_len)
    val_dataset = TinyStoriesDataset(str(cache_dir / "validation.bin"), seq_len=seq_len)
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **kwargs)
    return train_loader, val_loader
