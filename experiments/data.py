"""
Data pipeline for the optimizer benchmark.

Primary corpus: TinyStories (Eldan & Li, 2023).  It is small enough to train a
small transformer to a meaningful loss on a single Apple-silicon machine, and
unlike character-level corpora it exercises the embedding/attention/MLP
matrices that matrix orthogonalizers are designed for.

Fallback corpus: TinyShakespeare (character level), used for the
throughput/ablation runs where a fixed, tiny, fully local corpus is preferable.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    import urllib.request
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


# ---------------------------------------------------------------------------
# TinyStories (tokenized with a byte-level fallback: we train a tiny BPE-free
# character/byte model so the pipeline has zero external model dependency).
# ---------------------------------------------------------------------------

def prepare_tinystories(max_bytes: int | None = None):
    """
    Returns (train_ids, val_ids, vocab_size) as uint16 numpy arrays using a
    byte-level vocabulary (256 symbols).  Byte-level keeps the pipeline
    dependency-free and reproducible; the optimizer comparison is unaffected
    because all methods see exactly the same tokens.
    """
    base = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    files = {
        "train": "TinyStoriesV2-GPT4-train.txt",
        "valid": "TinyStoriesV2-GPT4-valid.txt",
    }
    out = {}
    for split, fn in files.items():
        p = DATA_DIR / fn
        try:
            _download(base + fn, p)
        except Exception as e:  # pragma: no cover
            print(f"  download failed for {fn}: {e}")
            return None
        raw = p.read_bytes()
        if max_bytes is not None:
            raw = raw[:max_bytes]
        out[split] = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
        print(f"  {split}: {out[split].size/1e6:.2f} M bytes")
    return out["train"], out["valid"], 256


def prepare_shakespeare():
    """Character-level TinyShakespeare, fully self-contained."""
    p = DATA_DIR / "tinyshakespeare.txt"
    _download("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
              "data/tinyshakespeare/input.txt", p)
    text = p.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def get_batch(data: np.ndarray, block: int, batch: int, device, generator=None):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
