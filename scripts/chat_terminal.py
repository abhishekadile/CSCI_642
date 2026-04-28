#!/usr/bin/env python3
"""
Terminal chat with the trained TinyStories model.
Supports: CUDA (RTX 2080 local), T4 (Colab), CPU fallback.
"""

import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokenizer import TinyStoriesTokenizer
from inference.chat import ChatSession
from model.transformer import GPTModel


def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--kv_mode", default="full", choices=["full", "sliding_window", "none"])
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    ckpt = safe_torch_load(args.checkpoint, device)
    config = ckpt.get("config") if isinstance(ckpt, dict) and ckpt.get("config") is not None else OmegaConf.load(args.config)
    model_config = config.model if hasattr(config, "model") else config["model"]
    model = GPTModel(model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    tokenizer = TinyStoriesTokenizer()

    session = ChatSession(
        model,
        tokenizer,
        device=str(device),
        kv_cache_mode=args.kv_mode,
        window_size=args.window_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    print("Chat started. Commands: 'quit' | 'reset' | 'save <filename>'")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.lower() == "reset":
            session.reset()
            print("[Conversation cleared]")
            continue
        if user_input.lower().startswith("save "):
            path = user_input[5:].strip()
            session.save_conversation(path)
            print(f"[Saved to {path}]")
            continue

        print("Model: ", end="", flush=True)
        response = session.chat(user_input)
        for char in response:
            print(char, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
