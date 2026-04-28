"""GPT-2 BPE tokenizer wrapper for TinyStories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable


class TinyStoriesTokenizer:
    """
    Custom BPE tokenizer for the TinyStories project.

    Wraps tiktoken's GPT-2 encoding with a HuggingFace fallback. The saved
    metadata file lets repeated Colab runs validate the tokenizer setup without
    redownloading HuggingFace assets when the cache directory is persisted.
    """

    VOCAB_SIZE = 50257
    EOS_TOKEN_ID = 50256
    PAD_TOKEN_ID = 50256

    def __init__(self, cache_dir: str = "data/tokenizer"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "tokenizer_meta.json"
        self.backend_name = "tiktoken"
        self._encoding = None
        self._hf_tokenizer = None

        if self.meta_path.exists():
            self._validate_metadata()

        try:
            import tiktoken

            self._encoding = tiktoken.get_encoding("gpt2")
            self.backend_name = "tiktoken"
        except Exception:
            from transformers import GPT2TokenizerFast

            self._hf_tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", cache_dir=str(self.cache_dir)
            )
            self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token
            self.backend_name = "transformers"

        self._write_metadata()

    def _validate_metadata(self) -> None:
        with self.meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("vocab_size") != self.VOCAB_SIZE:
            raise ValueError(
                f"Tokenizer metadata vocab mismatch: {meta.get('vocab_size')} != {self.VOCAB_SIZE}"
            )
        if meta.get("eos_token_id") != self.EOS_TOKEN_ID:
            raise ValueError(
                f"Tokenizer metadata EOS mismatch: {meta.get('eos_token_id')} != {self.EOS_TOKEN_ID}"
            )

    def _write_metadata(self) -> None:
        meta = {
            "type": "gpt2_bpe",
            "backend": self.backend_name,
            "vocab_size": self.VOCAB_SIZE,
            "eos_token_id": self.EOS_TOKEN_ID,
            "pad_token_id": self.PAD_TOKEN_ID,
        }
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def encode(self, text: str, add_eos: bool = False) -> list[int]:
        """Encode text to token IDs. Optionally append EOS."""
        if self._encoding is not None:
            ids = self._encoding.encode(text, allowed_special={"<|endoftext|>"})
        else:
            ids = self._hf_tokenizer.encode(text, add_special_tokens=False)
        if add_eos:
            ids.append(self.EOS_TOKEN_ID)
        return ids

    def decode(self, token_ids: Iterable[int], skip_special: bool = True) -> str:
        """Decode token IDs back to text."""
        ids = list(token_ids)
        if skip_special:
            ids = [token_id for token_id in ids if token_id != self.EOS_TOKEN_ID]
        if self._encoding is not None:
            return self._encoding.decode(ids)
        return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special)

    def encode_story(self, story: str) -> list[int]:
        """Encode a full story with EOS appended."""
        return self.encode(story, add_eos=True)

    def length_bin(self, token_ids: list[int]) -> str:
        """
        Bin story by token count for RQ2 stratification.
        Returns: 'short' (<100), 'medium' (100-250), 'long' (>250)
        """
        n = len(token_ids)
        if n < 100:
            return "short"
        if n <= 250:
            return "medium"
        return "long"

    @property
    def vocab_size(self) -> int:
        return self.VOCAB_SIZE

    def save(self, path: str) -> None:
        """Save tokenizer metadata, not BPE weights."""
        target = Path(path)
        if target.suffix:
            target.parent.mkdir(parents=True, exist_ok=True)
            out_path = target
        else:
            target.mkdir(parents=True, exist_ok=True)
            out_path = target / "tokenizer_meta.json"
        meta = {
            "type": "gpt2_bpe",
            "backend": self.backend_name,
            "vocab_size": self.VOCAB_SIZE,
            "eos_token_id": self.EOS_TOKEN_ID,
            "pad_token_id": self.PAD_TOKEN_ID,
            "cache_dir": str(self.cache_dir),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TinyStoriesTokenizer":
        """Load from saved metadata."""
        source = Path(path)
        meta_path = source if source.suffix else source / "tokenizer_meta.json"
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        cache_dir = meta.get("cache_dir") or os.fspath(meta_path.parent)
        return cls(cache_dir=cache_dir)
