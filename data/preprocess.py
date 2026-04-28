#!/usr/bin/env python3
"""Download TinyStories, tokenize with GPT-2 BPE, and cache binary tensors."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.tokenizer import TinyStoriesTokenizer
from utils.logging_utils import log


def cfg_get(config: Any, section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)


def tokenize_split(dataset, tokenizer: TinyStoriesTokenizer, split_name: str, out_path: Path) -> None:
    if out_path.exists():
        log(f"{out_path} exists; skipping {split_name} preprocessing")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    texts = [row.get("text") or row.get("story") or "" for row in dataset]
    total_tokens = 0

    with tmp_path.open("ab") as f:
        if tokenizer._encoding is not None:
            batch_size = 10_000
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Tokenizing {split_name}"):
                batch = texts[i : i + batch_size]
                encoded_batch = tokenizer._encoding.encode_batch(
                    batch, allowed_special={"<|endoftext|>"}
                )
                for ids in encoded_batch:
                    ids.append(tokenizer.EOS_TOKEN_ID)
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    total_tokens += len(ids)
        else:
            batch_size = 1_000
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Tokenizing {split_name}"):
                batch = texts[i : i + batch_size]
                encoded_batch = tokenizer._hf_tokenizer(
                    batch,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"]
                for ids in encoded_batch:
                    ids.append(tokenizer.EOS_TOKEN_ID)
                    np.asarray(ids, dtype=np.uint16).tofile(f)
                    total_tokens += len(ids)

    tmp_path.replace(out_path)
    log(f"Wrote {total_tokens:,} tokens to {out_path}")


def save_continuation_stories(dataset, tokenizer: TinyStoriesTokenizer, out_path: Path, n_stories: int) -> None:
    if out_path.exists():
        log(f"{out_path} exists; skipping continuation story cache")
        return
    rows = []
    for row in tqdm(dataset.select(range(min(n_stories, len(dataset)))), desc="Caching continuations"):
        story = row.get("text") or row.get("story") or ""
        tokens = tokenizer.encode_story(story)
        rows.append({"text": story, "tokens": tokens, "length_bin": tokenizer.length_bin(tokens)})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f)


def main() -> None:
    config_path = ROOT / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tokenizer = TinyStoriesTokenizer(cfg_get(config, "data", "tokenizer_cache_dir", "data/tokenizer"))
    cache_dir = ROOT / cfg_get(config, "data", "tensor_cache_dir", "data/cache")
    dataset_name = cfg_get(config, "data", "dataset_name", "roneneldan/TinyStories")
    n_cont = int(cfg_get(config, "data", "val_continuation_n_stories", 500))

    log(f"Loading {dataset_name}")
    ds = load_dataset(dataset_name)
    train_split = ds["train"]
    val_split_name = "validation" if "validation" in ds else "train"
    val_split = ds[val_split_name]
    if val_split_name == "train":
        val_split = val_split.select(range(min(10000, len(val_split))))

    tokenize_split(train_split, tokenizer, "train", cache_dir / "train.bin")
    tokenize_split(val_split, tokenizer, "validation", cache_dir / "validation.bin")
    save_continuation_stories(val_split, tokenizer, cache_dir / "continuation_stories.json", n_cont)


if __name__ == "__main__":
    main()
