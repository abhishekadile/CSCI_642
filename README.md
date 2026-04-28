# Efficient Small Transformer Training on TinyStories
### CSCI 642 — Natural Language Processing — Spring 2026 — Group 1
**Naomie Bambara · Abhishek Adile · Abdellah Afellah**

A decoder-only GPT-style Transformer trained on the TinyStories dataset with GPT-2 BPE tokenization, mixed precision training (AMP), KV caching, and GPU optimization. Supports training on Google Colab T4 and local inference on Windows with an RTX 2080 (CUDA).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Quickstart: Local Setup](#2-quickstart-local-setup)
3. [GitHub Repository](#3-github-repository)
4. [Train on Google Colab (Full Walkthrough)](#4-train-on-google-colab-full-walkthrough)
5. [Chat with the Model on Colab](#5-chat-with-the-model-on-colab)
6. [Chat from Your Windows Terminal (RTX 2080)](#6-chat-from-your-windows-terminal-rtx-2080)
7. [Checkpoint Strategy](#7-checkpoint-strategy)
8. [GPU Optimization Details](#8-gpu-optimization-details)
9. [Experiments (RQ1 and RQ2)](#9-experiments-rq1-and-rq2)
10. [Configuration Reference](#10-configuration-reference)
11. [Repository Map](#11-repository-map)

---

## 1. Project Overview

This project trains a compact GPT-style Transformer on TinyStories and investigates how sliding window KV cache sizes affect inference latency, GPU memory, and generation quality. The two research questions are:

**RQ1:** How do KV cache window sizes (64, 128, 256, full) affect latency, memory, and story quality in a compact decoder-only model?

**RQ2:** Does KV cache reuse efficiency (cache hit rate) correlate with lower continuation perplexity on story completion tasks?

---

## 2. Quickstart: Local Setup

**Requirements:** Python 3.10+, Git, CUDA 11.8+ (for RTX 2080), Windows or Linux.

```bash
# Clone the repo
git clone https://github.com/abhishekadile/CSCI_642.git
cd CSCI_642

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Preprocess the dataset (downloads TinyStories from HuggingFace, tokenizes, chunks)
# This runs once and caches binary tensors to data/cache/
python data/preprocess.py

# Verify your GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA GeForce RTX 2080
```

**Optional: run a quick local training smoke test (5 minutes)**

```bash
python scripts/train.py --time_limit 300 --device cuda
```

This trains for 5 minutes and saves a checkpoint to `checkpoints/`. Good for verifying the setup before a full training run.

---

## 3. GitHub Repository

```bash
# Clone the published project
git clone https://github.com/abhishekadile/CSCI_642.git
cd CSCI_642

# Pull the latest changes later
git pull origin main
```

Repository URL: https://github.com/abhishekadile/CSCI_642.git

Checkpoints, preprocessed tensors, results, tokenizer metadata, and wandb logs are excluded from git via `.gitignore`.

---

## 4. Train on Google Colab (Full Walkthrough)

Open `notebooks/train_colab.ipynb` in Google Colab. Run cells in order.

**Before you start:**
- In Colab, go to Runtime → Change runtime type → Hardware accelerator → T4 GPU
- Sign in to Google Drive when prompted (checkpoints will be saved there)

### Cell 1: Verify GPU

```python
!nvidia-smi
import torch
print(torch.cuda.get_device_name(0))
# Expected: Tesla T4
```

### Cell 2: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/tinystories_checkpoints', exist_ok=True)
os.environ['GDRIVE_PATH'] = '/content/drive/MyDrive/tinystories_checkpoints'
```

### Cell 3: Clone the Repository

```python
!git clone https://github.com/abhishekadile/CSCI_642.git
%cd CSCI_642
```

### Cell 4: Install Dependencies

```python
!pip install -r requirements.txt
```

### Cell 5: Preprocess Dataset

This downloads TinyStories from HuggingFace (~500MB), tokenizes with the GPT-2 BPE tokenizer, chunks into 256-token sequences, and saves binary tensor files to `data/cache/`. Takes 5–8 minutes on the first run. Results are cached — do not re-run unless you change chunk size.

```python
!python data/preprocess.py
```

### Cell 6: Start Training (1 Hour)

The training loop runs for exactly 1 hour (3600 seconds wall-clock), then saves the final checkpoint and exits cleanly. Checkpoints are saved to `checkpoints/` locally and synced to Google Drive every 500 steps, so a Colab disconnect will not lose your progress.

```python
!python scripts/train.py \
  --config configs/default.yaml \
  --time_limit 3600 \
  --device cuda
```

**What you will see during training:**

```
Step 100 | Loss: 4.231 | Val PPL: — | LR: 1.2e-4 | GPU: 8.4 GB / 15.9 GB
Step 500 | Loss: 3.187 | Val PPL: 24.3 | LR: 3e-4 | GPU: 9.1 GB / 15.9 GB | ✓ best checkpoint saved
Step 1000 | Loss: 2.844 | Val PPL: 17.8 | LR: 2.9e-4 | GPU: 9.1 GB / 15.9 GB | ✓ best checkpoint saved
...
[1:00:00] Time limit reached. Saving latest checkpoint. Training complete.
```

### Cell 7: Plot Training Curves

```python
import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('results/training_log.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(log['step'], log['train_loss'], label='Train Loss')
ax1.plot(log[log['val_loss'].notna()]['step'], log[log['val_loss'].notna()]['val_loss'], label='Val Loss')
ax1.set_xlabel('Step'); ax1.set_ylabel('Loss'); ax1.legend()
ax2.plot(log[log['val_ppl'].notna()]['step'], log[log['val_ppl'].notna()]['val_ppl'])
ax2.set_xlabel('Step'); ax2.set_ylabel('Validation Perplexity')
plt.tight_layout(); plt.show()
```

### Cell 8: Resume Training (if Colab disconnected)

If your session dropped, reconnect, re-run Cells 1–5, then:

```python
!python scripts/train.py \
  --config configs/default.yaml \
  --time_limit 3600 \
  --resume checkpoints/latest.pt \
  --device cuda
```

Training resumes from the saved model, optimizer, and AMP scaler state, then runs for the requested wall-clock time.

---

## 5. Chat with the Model on Colab

### Option A: Quick chat cell inside `train_colab.ipynb`

After training, add a new cell:

```python
from inference.chat import ChatSession
from model.transformer import GPTModel
from data.tokenizer import TinyStoriesTokenizer
import torch, yaml

config = yaml.safe_load(open('configs/default.yaml'))
tokenizer = TinyStoriesTokenizer()
model = GPTModel(config['model'])
ckpt = torch.load('checkpoints/best.pt', map_location='cuda')
model.load_state_dict(ckpt['model_state_dict'])
model = model.cuda().eval()

session = ChatSession(model, tokenizer, device='cuda')

# Chat loop — run this cell repeatedly
user_input = input("You: ")
response = session.chat(user_input)
print(f"Model: {response}")
```

### Option B: `chat_colab.ipynb`

Open `notebooks/chat_colab.ipynb`. It installs dependencies, loads `checkpoints/best.pt`, creates a `ChatSession`, and prompts for a single chat turn inside the notebook.

---

## 6. Chat from Your Windows Terminal (RTX 2080)

After training on Colab, download `checkpoints/best.pt` from Google Drive to your local machine.

```bash
# Activate your venv first
venv\Scripts\activate

# Start the chat REPL
python scripts/chat_terminal.py --checkpoint checkpoints/best.pt --device cuda
```

You will see:

```
Using device: NVIDIA GeForce RTX 2080 (8.6 GB)
Chat started. Commands: 'quit' | 'reset' | 'save <filename>'

You: Tell me a story about a robot who learns to bake bread.
Model: Once upon a time, there was a small robot named Benny who lived in a cozy kitchen...

You: What happens next?
Model: Benny carefully measured the flour, but he had never felt soft dough before...

You: save conversation.txt
[Saved to conversation.txt]

You: quit
Goodbye.
```

**Available commands in the terminal chat:**
- `quit` or `exit` — exit the program
- `reset` — clear conversation history and start fresh
- `save <filename>` — save the full conversation to a text file
- `--kv_mode sliding_window --window_size 128` — CLI flags to change cache mode

---

## 7. Checkpoint Strategy

Two checkpoints are maintained at all times:

| File | When Saved | Contents |
|---|---|---|
| `checkpoints/best.pt` | Every time validation loss improves | model weights with lowest observed val loss |
| `checkpoints/latest.pt` | Every 500 steps and at training end | most recent weights for resuming |

Each checkpoint file contains: `model_state_dict`, `optimizer_state_dict`, `scaler_state_dict` (for AMP), `step`, `val_loss`, `config`, and `timestamp`.

If `GDRIVE_PATH` is set, both checkpoints are copied to Google Drive after each save. This protects against Colab disconnection.

---

## 8. GPU Optimization Details

The following optimizations are applied automatically at training start:

| Optimization | Description | Benefit |
|---|---|---|
| AMP (FP16 + FP32) | Forward pass in FP16, master weights in FP32 | ~2x memory reduction, ~2x throughput |
| Dynamic loss scaling | Prevents FP16 underflow during backprop | Stable training at low precision |
| `cudnn.benchmark = True` | Auto-tunes CUDA kernels for fixed input shapes | 5–15% throughput gain |
| `allow_tf32 = True` | Uses TF32 on Ampere+ (RTX 2080 is Turing — approximation only) | Minor speedup |
| Pinned memory DataLoader | `pin_memory=True`, `prefetch_factor=2` | Faster host-to-device transfers |
| `torch.compile` | JIT-compiles the model graph (PyTorch 2.x) | 10–30% speedup where supported |
| Gradient accumulation | Accumulates gradients over 4 steps | Effective batch size 4x without extra VRAM |
| BatchSizeAutoTuner | Binary search for max batch size at startup | Maximizes GPU occupancy automatically |
| Cache flushing | `torch.cuda.empty_cache()` every 100 steps | Prevents memory fragmentation |

---

## 9. Experiments (RQ1 and RQ2)

After training, run the full experiment suite:

```bash
python scripts/run_experiments.py --checkpoint checkpoints/best.pt --skip_gpt4_eval
```

Results are saved to `results/rq1_results.csv` and `results/rq2_results.csv`.

**RQ1 output example:**

```
mode,window_size,latency_ms_per_token,peak_gpu_memory_mb,cache_hit_rate,gpt4_eval_skipped
none,0,18.4,1024,0.0,True
full,0,6.2,3840,0.94,True
sliding_window,64,7.4,896,0.94,True
sliding_window,128,7.1,1280,0.94,True
sliding_window,256,6.8,2048,0.94,True
```

**RQ2 output example:**

```
length_bin,n,pearson_r,p_value
all,500,-0.4200,0.0008
short,110,-0.3100,0.014
medium,260,-0.4500,0.0002
long,130,-0.5100,0.0001
```

---

## 10. Configuration Reference

All settings live in `configs/default.yaml`. Key parameters:

```yaml
model:
  n_layers: 6         # Transformer depth
  n_heads: 6          # Attention heads
  d_model: 384        # Hidden dimension (must be divisible by n_heads)
  max_seq_len: 256    # Training and inference context length

training:
  time_limit_seconds: 3600   # Wall-clock training budget (1 hour)
  val_every_steps: 500        # How often to compute validation perplexity
  grad_accumulation_steps: 4  # Effective batch = batch_size * grad_accumulation_steps

inference:
  default_kv_mode: "full"     # full | sliding_window | none
  default_window_size: 256    # Only used when kv_mode = sliding_window
  temperature: 0.8
  top_k: 50
```

---

## 11. Repository Map

```
CSCI_642/
│
├── .cursorrules          ← Cursor codebase map and vibe-coding rules
├── configs/default.yaml  ← All hyperparameters (single source of truth)
│
├── data/
│   ├── tokenizer.py      ← TinyStoriesTokenizer wrapper around GPT-2 BPE
│   ├── dataset.py        ← Memory-mapped binary token dataset
│   └── preprocess.py     ← Offline preprocessing → binary tensor cache
│
├── model/
│   ├── transformer.py    ← Full GPT decoder-only Transformer
│   ├── attention.py      ← Multi-head causal attention with KV cache interface
│   └── kv_cache.py       ← KVCache class: full + sliding window; hit rate logging
│
├── training/
│   ├── trainer.py        ← AMP training loop, cosine LR, grad clipping, val loop
│   ├── gpu_optimizer.py  ← CUDA optimizations, batch size auto-tuner, memory logger
│   └── checkpointing.py  ← Save best + latest; Google Drive sync; resume support
│
├── inference/
│   ├── generate.py       ← Autoregressive generation with all KV cache modes
│   └── chat.py           ← Multi-turn chat session with history management
│
├── evaluation/
│   ├── metrics.py        ← Perplexity, continuation perplexity, Pearson correlation
│   └── rq_experiments.py ← RQ1 and RQ2 full experiment sweeps
│
├── notebooks/
│   ├── train_colab.ipynb ← Full Colab training walkthrough (this README in notebook form)
│   └── chat_colab.ipynb  ← Inference-only Colab chat notebook
│
├── scripts/
│   ├── train.py             ← CLI training entry point
│   ├── chat_terminal.py     ← Windows terminal chat REPL with streaming output
│   └── run_experiments.py   ← RQ1 + RQ2 experiment runner
│
└── utils/
    ├── logging_utils.py  ← Console logging and CSV append helpers
    └── seed.py           ← Global reproducibility seed setter
```

---

## Reproducibility

Set `training.seed: 42` in `configs/default.yaml` (default). All entry point scripts call `set_seed()` before any data loading or model initialization. Results should be deterministic across identical hardware.

---

## References

- Vaswani et al. (2017). Attention Is All You Need. NeurIPS.
- Kaplan et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
- Eldan & Li (2023). TinyStories. arXiv:2305.07759.
- Micikevicius et al. (2018). Mixed Precision Training. ICLR 2018.
- Jiang et al. (2023). Mistral 7B. arXiv:2310.06825.
