"""Generation helpers."""

from __future__ import annotations

from typing import Optional

import torch

from model.kv_cache import KVCache


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    kv_cache_mode: str = "full",
    window_size: int = 256,
) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    kv_cache = None
    if kv_cache_mode != "none":
        kv_cache = KVCache(kv_cache_mode, window_size, n_layers=getattr(model, "n_layers", 6))
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.EOS_TOKEN_ID,
        kv_cache=kv_cache,
    )
    return tokenizer.decode(generated[0].tolist()[input_ids.size(1) :])
