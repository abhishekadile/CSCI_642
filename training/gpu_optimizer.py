"""GPU optimization utilities for training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from utils.logging_utils import log


def cfg_get(config: Any, section: str, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(section, {}).get(key, default)
    section_obj = getattr(config, section, None)
    return getattr(section_obj, key, default) if section_obj is not None else default


def cfg_set(config: Any, section: str, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config.setdefault(section, {})[key] = value
    else:
        setattr(getattr(config, section), key, value)


def setup_gpu_optimization(model, config, device):
    """
    Apply GPU optimizations. Returns the optimized model, possibly torch.compile'd.
    """
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if cfg_get(config, "gpu", "use_torch_compile", True):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            log("torch.compile applied (reduce-overhead mode)")
        except Exception as exc:
            log(f"torch.compile not available: {exc}. Skipping.")
    return model


class BatchSizeAutoTuner:
    """Binary search for the largest batch size that fits in GPU memory."""

    def __init__(self, model, config, device, max_trials: int = 6):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.max_trials = max_trials
        self.seq_len = int(cfg_get(config, "data", "chunk_size", cfg_get(config, "model", "max_seq_len", 256)))
        self.vocab_size = int(cfg_get(config, "model", "vocab_size", 50257))
        self.upper = int(cfg_get(config, "training", "batch_size", 32))

    def _fits(self, batch_size: int) -> bool:
        if self.device.type != "cuda":
            return True
        try:
            torch.cuda.empty_cache()
            self.model.train()
            x = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
            y = torch.randint(0, self.vocab_size, (batch_size, self.seq_len), device=self.device)
            out = self.model(x, targets=y)
            out["loss"].backward()
            self.model.zero_grad(set_to_none=True)
            allocated = torch.cuda.memory_allocated(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            del x, y, out
            torch.cuda.empty_cache()
            return allocated < 0.90 * total
        except RuntimeError as exc:
            self.model.zero_grad(set_to_none=True)
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                return False
            raise

    def tune(self) -> int:
        if self.device.type != "cuda":
            log("Batch auto-tune skipped on non-CUDA device")
            return self.upper
        low, high = 1, self.upper
        best = 1
        trials = 0
        while low <= high and trials < self.max_trials:
            mid = (low + high) // 2
            trials += 1
            if self._fits(mid):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
            if high - low < 2 and trials >= 2:
                break
        cfg_set(self.config, "training", "batch_size", best)
        log(f"Auto-tuned batch size: {best}")
        return best


def create_optimized_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int = 4) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def log_memory_snapshot(step: int, csv_path: str = "results/memory_log.csv") -> None:
    if not torch.cuda.is_available():
        return
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["step", "allocated_mb", "reserved_mb", "max_allocated_mb"])
        writer.writerow(
            [
                step,
                torch.cuda.memory_allocated() / 1024**2,
                torch.cuda.memory_reserved() / 1024**2,
                torch.cuda.max_memory_allocated() / 1024**2,
            ]
        )


def empty_cache_periodically(step: int, every_n: int = 100) -> None:
    if torch.cuda.is_available() and step % every_n == 0:
        torch.cuda.empty_cache()
