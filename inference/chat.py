"""Multi-turn chat wrapper around TinyStories generation."""

from __future__ import annotations

from pathlib import Path

import torch

from model.kv_cache import KVCache


class ChatSession:
    """
    Multi-turn conversational wrapper over the generation engine.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "auto",
        kv_cache_mode: str = "full",
        window_size: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        max_new_tokens: int = 200,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.kv_cache_mode = kv_cache_mode
        self.window_size = window_size
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = getattr(model, "max_seq_len", 256)
        self.history_tokens: list[int] = []
        self.turns: list[tuple[str, str]] = []

        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            print(f"Using device: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print("Using device: CPU")

    def chat(self, user_input: str) -> str:
        """Single turn: encode input, generate, return response string."""
        prompt = f"\nUser: {user_input}\nModel:"
        user_tokens = self.tokenizer.encode(prompt)
        self.history_tokens.extend(user_tokens)
        self.history_tokens = self.history_tokens[-self.max_seq_len :]

        input_ids = torch.tensor([self.history_tokens], dtype=torch.long, device=self.device)
        kv_cache = None
        if self.kv_cache_mode != "none":
            kv_cache = KVCache(self.kv_cache_mode, self.window_size, n_layers=getattr(self.model, "n_layers", 6))

        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                eos_token_id=self.tokenizer.EOS_TOKEN_ID,
                kv_cache=kv_cache,
            )

        new_tokens = generated[0].tolist()[input_ids.size(1) :]
        response = self.tokenizer.decode(new_tokens).split("\nUser:")[0].strip()
        response_tokens = self.tokenizer.encode(response, add_eos=False)
        self.history_tokens.extend(response_tokens)
        self.history_tokens = self.history_tokens[-self.max_seq_len :]
        self.turns.append((user_input, response))
        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self.history_tokens = []
        self.turns = []

    def save_conversation(self, path: str) -> None:
        """Save full conversation to a plain text file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True) if out_path.parent != Path(".") else None
        with out_path.open("w", encoding="utf-8") as f:
            for user, model in self.turns:
                f.write(f"You: {user}\nModel: {model}\n\n")
