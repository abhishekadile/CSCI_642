"""Small GPT-style transformer for TinyStories."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.kv_cache import KVCache


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm * (x.shape[-1] ** -0.5)
        return self.weight * x / (rms + self.eps)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        use_rms_norm: bool,
        layer_idx: int,
    ):
        super().__init__()
        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
        self.layer_idx = layer_idx
        self.norm1 = norm_cls(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = norm_cls(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), kv_cache=kv_cache, layer_idx=self.layer_idx)
        x = x + self.ff(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.n_layers = int(cfg_get(config, "n_layers", 6))
        self.n_heads = int(cfg_get(config, "n_heads", 6))
        self.d_model = int(cfg_get(config, "d_model", 384))
        self.d_ff = int(cfg_get(config, "d_ff", 1536))
        self.max_seq_len = int(cfg_get(config, "max_seq_len", 256))
        self.vocab_size = int(cfg_get(config, "vocab_size", 50257))
        self.dropout_p = float(cfg_get(config, "dropout", 0.1))
        self.use_rms_norm = bool(cfg_get(config, "use_rms_norm", True))

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    self.dropout_p,
                    self.use_rms_norm,
                    layer_idx=i,
                )
                for i in range(self.n_layers)
            ]
        )
        norm_cls = RMSNorm if self.use_rms_norm else nn.LayerNorm
        self.final_norm = norm_cls(self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)

        if bool(cfg_get(config, "weight_tying", True)):
            self.output_projection.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len and kv_cache is None:
            input_ids = input_ids[:, -self.max_seq_len :]
            if targets is not None:
                targets = targets[:, -self.max_seq_len :]
            seq_len = input_ids.shape[1]

        if kv_cache is not None:
            cached = kv_cache.get(0)
            past_len = cached[0].size(2) if cached is not None else 0
            pos = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            pos = torch.clamp(pos, max=self.max_seq_len - 1)
        else:
            pos = torch.arange(seq_len, device=input_ids.device)

        x = self.token_embedding(input_ids) + self.position_embedding(pos).unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, kv_cache=kv_cache)
        logits = self.output_projection(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        repetition_penalty: float = 1.0,
        eos_token_id: int = 50256,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = input_ids
        next_input = input_ids[:, -self.max_seq_len :] if kv_cache is not None else input_ids

        for _ in range(max_new_tokens):
            model_input = next_input if kv_cache is not None else generated[:, -self.max_seq_len :]
            logits = self(model_input, kv_cache=kv_cache)["logits"][:, -1, :]

            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[:, token_id] /= repetition_penalty

            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                remove = cumulative_probs > top_p
                remove[:, 1:] = remove[:, :-1].clone()
                remove[:, 0] = False
                sorted_logits[remove] = float("-inf")
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            next_input = next_token
            if next_token.item() == eos_token_id:
                break
        return generated

    def summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        size_fp32_mb = total * 4 / 1024**2
        size_fp16_mb = total * 2 / 1024**2

        print(f"{'-' * 45}")
        print("  Model Summary")
        print(f"{'-' * 45}")
        print(f"  Total parameters:      {total:>12,}")
        print(f"  Trainable parameters:  {trainable:>12,}")
        print(f"  Size (FP32):           {size_fp32_mb:>11.1f} MB")
        print(f"  Size (FP16/AMP):       {size_fp16_mb:>11.1f} MB")
        print(f"{'-' * 45}")

        return {
            "total_params": total,
            "trainable_params": trainable,
            "size_fp32_mb": size_fp32_mb,
            "size_fp16_mb": size_fp16_mb,
        }
