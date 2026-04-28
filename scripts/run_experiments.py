#!/usr/bin/env python3
"""Run RQ1/RQ2 experiment sweeps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokenizer import TinyStoriesTokenizer
from evaluation.rq_experiments import run_rq1, run_rq2
from model.transformer import GPTModel


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_gpt4_eval", action="store_true")
    parser.add_argument("--rq1", action="store_true")
    parser.add_argument("--rq2", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    ckpt = safe_torch_load(args.checkpoint, device)
    config = ckpt.get("config") if isinstance(ckpt, dict) and ckpt.get("config") is not None else OmegaConf.load(args.config)
    model_config = config.model if hasattr(config, "model") else config["model"]
    model = GPTModel(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = TinyStoriesTokenizer()

    if not args.rq1 and not args.rq2:
        args.rq1 = args.rq2 = True
    if args.rq1:
        run_rq1(model, tokenizer, config, device, skip_gpt4_eval=args.skip_gpt4_eval)
    if args.rq2:
        run_rq2(model, tokenizer, config, device)


if __name__ == "__main__":
    main()
