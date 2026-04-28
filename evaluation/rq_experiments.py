"""Research question experiment sweeps."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from evaluation.metrics import continuation_perplexity, pearson_r
from model.kv_cache import KVCache
from utils.logging_utils import log


RQ1_PROMPTS = [
    "Once upon a time, there was a little rabbit who",
    "Lily found a shiny key under her bed and",
    "The small robot wanted to learn how to",
    "Tom and Mia went into the forest because",
    "Every night, the moon told stories about",
] * 10


def cfg_get(config: Any, section: str, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(section, {}).get(key, default)
    section_obj = getattr(config, section, None)
    return getattr(section_obj, key, default) if section_obj is not None else default


def _write_rows(path: str, header: list[str], rows: list[dict]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def _n_layers(model, config) -> int:
    return int(cfg_get(config, "model", "n_layers", getattr(model, "n_layers", 6)))


@torch.no_grad()
def run_rq1(model, tokenizer, config, device, skip_gpt4_eval: bool = True):
    """
    RQ1: How do KV cache window sizes affect latency, GPU memory, and quality?
    """
    device = torch.device(device)
    model.eval()
    max_new = int(cfg_get(config, "inference", "max_new_tokens", 200))
    temp = float(cfg_get(config, "inference", "temperature", 0.8))
    top_k = int(cfg_get(config, "inference", "top_k", 50))
    top_p = float(cfg_get(config, "inference", "top_p", 0.95))
    n_layers = _n_layers(model, config)
    modes = [("none", 0), ("full", 0), ("sliding_window", 64), ("sliding_window", 128), ("sliding_window", 256)]
    rows = []

    for mode, window in modes:
        latencies = []
        peak_mems = []
        hit_rates = []
        for prompt in RQ1_PROMPTS[:50]:
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            kv_cache = None if mode == "none" else KVCache(mode, window or 256, n_layers=n_layers)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new,
                temperature=temp,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=tokenizer.EOS_TOKEN_ID,
                kv_cache=kv_cache,
            )
            elapsed = time.perf_counter() - start
            new_tokens = max(1, generated.size(1) - input_ids.size(1))
            latencies.append((elapsed * 1000) / new_tokens)
            peak_mems.append(torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0)
            hit_rates.append(kv_cache.cache_hit_rate if kv_cache is not None else 0.0)

        rows.append(
            {
                "mode": mode,
                "window_size": window,
                "latency_ms_per_token": sum(latencies) / len(latencies),
                "peak_gpu_memory_mb": max(peak_mems),
                "cache_hit_rate": sum(hit_rates) / len(hit_rates),
                "gpt4_eval_skipped": skip_gpt4_eval,
            }
        )
        log(f"RQ1 {mode}:{window} latency={rows[-1]['latency_ms_per_token']:.2f} ms/token")

    _write_rows(
        "results/rq1_results.csv",
        ["mode", "window_size", "latency_ms_per_token", "peak_gpu_memory_mb", "cache_hit_rate", "gpt4_eval_skipped"],
        rows,
    )
    return rows


@torch.no_grad()
def run_rq2(model, tokenizer, config, device):
    """
    RQ2: Does KV cache hit rate correlate with lower continuation perplexity?
    """
    device = torch.device(device)
    model.eval()
    cache_path = Path(cfg_get(config, "data", "tensor_cache_dir", "data/cache")) / "continuation_stories.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run `python data/preprocess.py` before `scripts/run_experiments.py`."
        )
    with cache_path.open("r", encoding="utf-8") as f:
        stories = json.load(f)

    max_seq_len = int(cfg_get(config, "model", "max_seq_len", 256))
    n_layers = _n_layers(model, config)
    rows = []
    for idx, row in enumerate(stories[: int(cfg_get(config, "data", "val_continuation_n_stories", 500))]):
        tokens = row.get("tokens") or tokenizer.encode_story(row["text"])
        if len(tokens) < 4:
            continue
        midpoint = len(tokens) // 2
        prompt_tokens = tokens[:midpoint]
        continuation_tokens = tokens[midpoint:]
        prompt_ids = torch.tensor([prompt_tokens[-max_seq_len:]], dtype=torch.long, device=device)
        kv_cache = KVCache("full", max_seq_len, n_layers=n_layers)
        _ = model.generate(
            prompt_ids,
            max_new_tokens=min(32, len(continuation_tokens)),
            temperature=float(cfg_get(config, "inference", "temperature", 0.8)),
            top_k=int(cfg_get(config, "inference", "top_k", 50)),
            top_p=float(cfg_get(config, "inference", "top_p", 0.95)),
            eos_token_id=tokenizer.EOS_TOKEN_ID,
            kv_cache=kv_cache,
        )
        ppl = continuation_perplexity(model, prompt_tokens, continuation_tokens, device, max_seq_len)
        rows.append(
            {
                "story_idx": idx,
                "cache_hit_rate": kv_cache.cache_hit_rate,
                "continuation_ppl": ppl,
                "length_bin": row.get("length_bin") or tokenizer.length_bin(tokens),
            }
        )

    _write_rows("results/rq2_results.csv", ["story_idx", "cache_hit_rate", "continuation_ppl", "length_bin"], rows)

    global_r, global_p = pearson_r([r["cache_hit_rate"] for r in rows], [r["continuation_ppl"] for r in rows])
    print("length_bin,n,pearson_r,p_value")
    print(f"all,{len(rows)},{global_r:.4f},{global_p:.4g}")
    for length_bin in ("short", "medium", "long"):
        subset = [r for r in rows if r["length_bin"] == length_bin]
        r, p = pearson_r([x["cache_hit_rate"] for x in subset], [x["continuation_ppl"] for x in subset])
        print(f"{length_bin},{len(subset)},{r:.4f},{p:.4g}")
    return rows
