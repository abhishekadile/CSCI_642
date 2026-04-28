"""Small logging helpers used by scripts and training."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def append_csv_row(path: str, header: list[str], row: list[object]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        if write_header:
            f.write(",".join(header) + "\n")
        f.write(",".join("" if v is None else str(v) for v in row) + "\n")
