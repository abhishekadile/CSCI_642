#!/usr/bin/env python3
"""Train the TinyStories GPT model locally or on Colab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import TinyStoriesDataset
from model.transformer import GPTModel
from training.checkpointing import load_checkpoint
from training.gpu_optimizer import (
    BatchSizeAutoTuner,
    create_optimized_dataloader,
    setup_gpu_optimization,
)
from training.trainer import train
from utils.logging_utils import log
from utils.seed import set_seed


def create_adamw(model, config, device):
    kwargs = {
        "lr": float(config.training.learning_rate),
        "betas": (0.9, 0.95),
    }
    use_fused = bool(config.gpu.get("use_fused_adamw", True))
    if device.type == "cuda" and use_fused:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
            log("Using fused AdamW optimizer")
            return optimizer
        except TypeError:
            log("Fused AdamW not available; falling back to standard AdamW")
    return torch.optim.AdamW(model.parameters(), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--time_limit", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.time_limit is not None:
        config.training.time_limit_seconds = args.time_limit
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    device = torch.device(device_name)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    set_seed(int(config.training.seed))
    model = GPTModel(config.model).to(device)
    model.summary()

    cache_dir = Path(config.data.tensor_cache_dir)
    seq_len = int(config.data.chunk_size)
    train_dataset = TinyStoriesDataset(
        str(cache_dir / "train.bin"),
        seq_len=seq_len,
    )
    val_dataset = TinyStoriesDataset(
        str(cache_dir / "validation.bin"),
        seq_len=seq_len,
    )

    if config.gpu.auto_tune_batch_size:
        BatchSizeAutoTuner(model, config, device).tune()

    train_loader = create_optimized_dataloader(
        train_dataset,
        int(config.training.batch_size),
        shuffle=True,
        num_workers=int(config.data.num_workers),
        prefetch_factor=int(config.data.get("prefetch_factor", 2)),
    )
    val_loader = create_optimized_dataloader(
        val_dataset,
        int(config.training.batch_size),
        shuffle=False,
        num_workers=int(config.data.num_workers),
        prefetch_factor=int(config.data.get("prefetch_factor", 2)),
    )

    optimizer = create_adamw(model, config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    resume_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, device=device)
        resume_step = int(ckpt.get("step", 0))

    model = setup_gpu_optimization(model, config, device)
    log(
        f"Starting training on {device} "
        f"with batch_size={config.training.batch_size}"
    )
    train(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        config,
        resume_step=resume_step,
        scaler=scaler,
    )


if __name__ == "__main__":
    main()
