"""AMP training loop with time limits and checkpointing."""

from __future__ import annotations

import math
import subprocess
import time
from itertools import cycle
from typing import Any

import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from training import checkpointing
from training.gpu_optimizer import (
    empty_cache_periodically,
    log_memory_snapshot,
)
from utils.logging_utils import append_csv_row, log


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def get_gpu_utilization(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first_line = result.stdout.strip().splitlines()[0]
    try:
        return int(first_line.strip())
    except ValueError:
        return None


def cfg_get(config: Any, section: str, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(section, {}).get(key, default)
    section_obj = getattr(config, section, None)
    if section_obj is None:
        return default
    return getattr(section_obj, key, default)


def cosine_lr(
    step: int,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    progress = min(
        1.0,
        (step - warmup_steps) / max(1, total_steps - warmup_steps),
    )
    return 0.5 * learning_rate * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_loader, device, config) -> tuple[float, float, int]:
    model.eval()
    losses = []
    max_batches = int(cfg_get(config, "training", "val_max_batches", 50))
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x, targets=y)
        losses.append(out["loss"].item())
    model.train()
    loss = sum(losses) / max(1, len(losses))
    return loss, math.exp(min(loss, 20)), len(losses)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    config,
    resume_step: int = 0,
    scaler=None,
) -> None:
    device = torch.device(device)
    scaler = scaler or GradScaler("cuda", enabled=device.type == "cuda")
    grad_accum = int(cfg_get(config, "training", "grad_accumulation_steps", 4))
    lr = float(cfg_get(config, "training", "learning_rate", 3e-4))
    warmup_steps = int(cfg_get(config, "training", "warmup_steps", 500))
    time_limit = int(cfg_get(config, "training", "time_limit_seconds", 3600))
    val_every = int(cfg_get(config, "training", "val_every_steps", 500))
    val_warmup = int(cfg_get(config, "training", "val_warmup_steps", 100))
    save_every = int(cfg_get(config, "training", "save_every_steps", 500))
    log_every = max(
        1,
        int(cfg_get(config, "training", "log_every_steps", 100)),
    )
    progress_style = str(cfg_get(config, "training", "progress_style", "line"))
    max_grad_norm = float(cfg_get(config, "training", "max_grad_norm", 1.0))
    empty_every = int(cfg_get(config, "gpu", "empty_cache_every_n_steps", 100))
    memory_every = int(cfg_get(config, "gpu", "memory_log_every_n_steps", 200))

    step = resume_step
    best_val_loss = float("inf")
    val_loss = None
    training_start = time.time()
    last_elapsed = 0.0
    last_log_time = training_start
    last_log_step = step
    total_steps_est = max(1, time_limit * max(1, len(train_loader)) // 3600)
    show_progress_bar = progress_style == "tqdm"
    pbar = (
        tqdm(total=time_limit, unit="s", desc="Training")
        if show_progress_bar
        else None
    )
    optimizer.zero_grad(set_to_none=True)

    if not show_progress_bar:
        log(f"Starting training for {format_duration(time_limit)}")
        log("=" * 70)

    try:
        for x, y in cycle(train_loader):
            elapsed = time.time() - training_start
            if elapsed >= time_limit:
                log("Time limit reached. Saving latest checkpoint.")
                checkpointing.save_latest(
                    model,
                    optimizer,
                    scaler,
                    step,
                    val_loss,
                    config,
                )
                break

            current_lr = cosine_lr(step, lr, warmup_steps, total_steps_est)
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x, targets=y)
                loss = out["loss"] / grad_accum

            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            now_elapsed = time.time() - training_start
            delta = max(0.0, now_elapsed - last_elapsed)
            if pbar is not None:
                remaining = max(0.0, time_limit - pbar.n)
                pbar.update(min(delta, remaining))
            last_elapsed = now_elapsed

            should_log = step % log_every == 0
            train_loss = None
            train_ppl = None
            gpu_mem_mb = None
            if should_log:
                train_loss = loss.detach().item() * grad_accum
                train_ppl = math.exp(min(train_loss, 20))
                gpu_mem_mb = (
                    torch.cuda.memory_allocated() / 1024**2
                    if device.type == "cuda"
                    else 0.0
                )
                if pbar is not None:
                    pbar.set_postfix(
                        {
                            "loss": f"{train_loss:.3f}",
                            "ppl": f"{train_ppl:.1f}",
                            "lr": f"{current_lr:.2e}",
                            "gpu_mb": f"{gpu_mem_mb:.0f}",
                        }
                    )
                else:
                    log_elapsed = max(1e-6, now_elapsed - last_log_time)
                    step_delta = max(1, step - last_log_step)
                    batch_size = int(
                        cfg_get(config, "training", "batch_size", 1)
                    )
                    seq_len = int(cfg_get(config, "data", "chunk_size", 1))
                    tok_s = int(
                        step_delta * batch_size * seq_len / log_elapsed
                    )
                    eta = max(0.0, time_limit - now_elapsed)
                    gpu_util = get_gpu_utilization(device)
                    gpu_text = (
                        f"{gpu_util}%"
                        if gpu_util is not None
                        else "n/a"
                    )
                    print(
                        f"Step {step:,} | Loss: {train_loss:.4f} | "
                        f"PPL: {train_ppl:.2f} | Tok/s: {tok_s:,} | "
                        f"LR: {current_lr:.2e} | "
                        f"Elapsed: {format_duration(now_elapsed)} | "
                        f"ETA: {format_duration(eta)} | GPU: {gpu_text}",
                        flush=True,
                    )
                    last_log_time = now_elapsed
                    last_log_step = step

            if step % memory_every == 0:
                log_memory_snapshot(step)
            empty_cache_periodically(step, empty_every)

            row_val_loss = None
            row_val_ppl = None
            if step > 0 and step % val_every == 0:
                if step < val_warmup:
                    log(f"Step {step}: skipping val eval (warm-up period)")
                else:
                    if not show_progress_bar:
                        print("\nEvaluating...", flush=True)
                    val_loss, val_ppl, eval_batches = evaluate(
                        model, val_loader, device, config
                    )
                    row_val_loss, row_val_ppl = val_loss, val_ppl
                    log(
                        f"Evaluated {eval_batches} batches | "
                        f"val_loss={val_loss:.4f}, val_ppl={val_ppl:.2f}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        checkpointing.save_best(
                            model,
                            optimizer,
                            scaler,
                            step,
                            val_loss,
                            config,
                        )

            if step > 0 and step % save_every == 0:
                checkpointing.save_latest(
                    model,
                    optimizer,
                    scaler,
                    step,
                    val_loss,
                    config,
                )

            if should_log or row_val_loss is not None:
                append_csv_row(
                    "results/training_log.csv",
                    [
                        "step",
                        "train_loss",
                        "train_ppl",
                        "val_loss",
                        "val_ppl",
                        "lr",
                        "gpu_mb",
                    ],
                    [
                        step,
                        train_loss,
                        train_ppl,
                        row_val_loss,
                        row_val_ppl,
                        current_lr,
                        gpu_mem_mb,
                    ],
                )
            step += 1
    except KeyboardInterrupt:
        log("Interrupted. Saving latest checkpoint before exit.")
        checkpointing.save_latest(
            model,
            optimizer,
            scaler,
            step,
            val_loss,
            config,
        )
    finally:
        if pbar is not None:
            pbar.close()
