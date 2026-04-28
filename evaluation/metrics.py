"""Evaluation metrics for TinyStories experiments."""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.stats import pearsonr


def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 20))


@torch.no_grad()
def continuation_perplexity(model, prompt_tokens: list[int], continuation_tokens: list[int], device, max_seq_len: int) -> float:
    all_tokens = (prompt_tokens + continuation_tokens)[-max_seq_len:]
    if len(all_tokens) < 2:
        return float("nan")
    x = torch.tensor([all_tokens[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([all_tokens[1:]], dtype=torch.long, device=device)
    out = model(x, targets=y)
    return compute_perplexity(float(out["loss"].item()))


def pearson_r(xs: list[float], ys: list[float]) -> tuple[float, float]:
    clean = [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y)]
    if len(clean) < 2:
        return float("nan"), float("nan")
    x_arr, y_arr = zip(*clean)
    if len(set(x_arr)) < 2 or len(set(y_arr)) < 2:
        return float("nan"), float("nan")
    r, p = pearsonr(x_arr, y_arr)
    return float(r), float(p)


def compute_distinct_n(texts: list[str], n: int = 2) -> float:
    total = 0
    unique = set()
    for text in texts:
        tokens = text.split()
        grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0
