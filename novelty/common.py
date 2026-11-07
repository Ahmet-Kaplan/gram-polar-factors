"""
Shared machinery for the novelty/ hypotheses.

Deliberately independent of `experiments/` and `large_scale/`: this module
builds its own corpus, model and momentum buffers so that no result reported
from here can be an inheritance from an earlier conclusion.

Conventions
-----------
A "momentum matrix" is one 2-D hidden weight of the transformer, oriented so
that rows <= columns (the shape the polar factor is defined for).  A "stream"
is one independent sequence of minibatches feeding an exponential moving
average of the gradient.  All spectra are computed in float64 on the CPU:
torch's MPS backend has no float64, and measuring a 1e-9 ratio in float32 is
meaningless by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
REPO = HERE.parent
DATA = HERE / "data"   # self-contained: this folder owns its corpus
CORPUS = DATA / "TinyStoriesV2-GPT4-train.txt"

RHO = 0.15           # row-subsampling fraction used by the optimizer we study
BETA = 0.95          # momentum coefficient
BATCH = 16
BLOCK = 128

# Gradients and training run on the accelerator in float32.  Spectra, polar
# factors and every ratio reported from this folder are float64: a ratio of
# 1e-9 cannot be measured in float32, and a float32 eigendecomposition of a
# matrix with that ratio is meaningless by construction.
DTYPE = torch.float32   # the dtype the model and its gradients live in
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPECTRAL = DEVICE if DEVICE.type == "cuda" else torch.device("cpu")


def describe() -> str:
    s = f"device={DEVICE} spectral={SPECTRAL} torch={torch.__version__}"
    if DEVICE.type == "cuda":
        s += (f" gpu={torch.cuda.get_device_name(0)}"
              f" cc={'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    return s


# ---------------------------------------------------------------------------
# corpus and model
# ---------------------------------------------------------------------------

def corpus(seed: int = 0, n_bytes: int = 40_000_000):
    """Byte-level token ids.  Deterministic; no tokenizer dependency."""
    raw = np.frombuffer(CORPUS.read_bytes()[:n_bytes], dtype=np.uint8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(raw) - 1)
    return raw, perm


@dataclass
class Cfg:
    vocab: int = 256
    block: int = BLOCK
    n_layer: int = 4
    n_head: int = 6
    n_embd: int = 192


class Attn(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c_attn = nn.Linear(c.n_embd, 3 * c.n_embd, bias=False)
        self.c_proj = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.h = c.n_head

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2)
                   for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


class MLP(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c_fc = nn.Linear(c.n_embd, 4 * c.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * c.n_embd, c.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attn(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = MLP(c)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab, c.n_embd)
        self.wpe = nn.Embedding(c.block, c.n_embd)
        self.h = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        for b in self.h:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def make_model(seed: int = 0, depth: int = 4, width: int = 192) -> GPT:
    torch.manual_seed(seed)
    m = GPT(Cfg(n_layer=depth, n_embd=width, n_head=max(2, width // 32)))
    return m.to(DEVICE)


def to64(B: torch.Tensor) -> torch.Tensor:
    return B.detach().to(SPECTRAL, torch.float64)


def hidden(model: GPT) -> dict[str, nn.Parameter]:
    """The matrices a matrix-orthogonalizer actually acts on.

    Only the 2-D weights inside the transformer blocks.  Embeddings and the
    output head are excluded: Muon-family optimizers leave them on AdamW, and
    including them would mix two different objects into one statistic.
    """
    out = {}
    for name, p in model.named_parameters():
        if not name.startswith("h.") or p.dim() != 2:
            continue
        out[name] = p
    return out


def orth_err(P: torch.Tensor) -> float:
    """||P P^T - I_m||_F / sqrt(m): the standard orthogonality defect.

    P is m x n with rows that should be orthonormal, so the projector is
    P P^T (m x m), not P^T P.
    """
    P = P.double()
    m = P.shape[0]
    I = torch.eye(m, device=P.device, dtype=P.dtype)
    return float((P @ P.T - I).norm() / m ** 0.5)


def thin(W: torch.Tensor) -> torch.Tensor:
    """Orient a weight so rows <= columns (the polar-factor convention)."""
    return W if W.shape[0] <= W.shape[1] else W.T


def batches(raw: np.ndarray, perm: np.ndarray, stream: int, n: int,
            batch: int = BATCH, block: int = BLOCK,
            start: int | None = None) -> list[tuple]:
    """`n` (x, y) blocks, either for stream `stream` or from block index `start`.

    `start` is in units of one batch, which lets a caller index the data
    explicitly instead of relying on the stream-times-length offset.
    """
    out = []
    base = stream * n * batch * block if start is None else start * batch * block
    for i in range(n):
        off = base + i * batch * block
        ix = perm[off: off + batch * block]
        x = torch.from_numpy(raw[ix].astype(np.int64)).view(batch, block)
        y = torch.from_numpy(raw[ix + 1].astype(np.int64)).view(batch, block)
        out.append((x.to(DEVICE), y.to(DEVICE)))
    return out


def gradients(model: GPT, x, y) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    _, loss = model(x, y)
    loss.backward()
    return {n: p.grad.detach().clone() for n, p in hidden(model).items()}


def momentum(model: GPT, raw, perm, stream: int, steps: int = 21,
             beta: float = BETA, buf_dtype: torch.dtype = torch.float32,
             autocast: bool = False, batch: int = BATCH,
             block: int = BLOCK) -> tuple[dict[str, torch.Tensor], float]:
    """EMA of the gradient at *fixed* weights over `steps` minibatches.

    `buf_dtype` is the dtype the buffer is accumulated in -- the thing H1 varies,
    since every implementation stores it in float32.  `autocast` runs the forward
    and backward in bfloat16, which is what a GPU training run does.
    """
    H = hidden(model)
    acc = {n: torch.zeros(p.shape, dtype=buf_dtype, device=p.device)
           for n, p in H.items()}
    losses = []
    for x, y in batches(raw, perm, stream, steps, batch, block):
        model.zero_grad(set_to_none=True)
        if autocast:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=(DEVICE.type == "cuda")):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        loss.backward()
        losses.append(float(loss.detach()))
        with torch.no_grad():
            for n, p in H.items():
                acc[n].mul_(beta).add_(p.grad.to(buf_dtype), alpha=1 - beta)
    return acc, float(np.mean(losses))


# ---------------------------------------------------------------------------
# spectra, polar factors, subsampling
# ---------------------------------------------------------------------------

def subsample(B: torch.Tensor, rho: float = RHO) -> torch.Tensor:
    """Keep the ceil(rho*m) rows of largest Euclidean norm (Dion3's rule)."""
    B = to64(B)
    m = B.shape[0]
    k = max(1, int(np.ceil(rho * m)))
    norms = B.norm(dim=1)
    idx = torch.topk(norms, k).indices
    return B[idx]


def polar(B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Polar factor U V^T and the singular values, in float64."""
    B = to64(B)
    U, s, Vh = torch.linalg.svd(B, full_matrices=False)
    return (U @ Vh), s


def svdvals(B: torch.Tensor) -> torch.Tensor:
    return torch.linalg.svdvals(to64(B))


def snapshot(model: GPT) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in hidden(model).items()}


def as64(D: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.double() for k, v in D.items()}


def summarize_kappa(D: dict[str, torch.Tensor], do_subsample: bool = True) -> dict:
    """Median sigma_min/sigma_max over the matrices, before and after subsampling."""
    full, sub = [], []
    for B in D.values():
        B = to64(B)
        if B.abs().max() == 0:
            continue
        s = torch.linalg.svdvals(B)
        full.append(float(s[-1] / s[0]))
        s2 = torch.linalg.svdvals(subsample(B))
        sub.append(float(s2[-1] / s2[0]))
    out = {"median_ratio_full": float(np.median(full)),
           "n_matrices": len(full)}
    if do_subsample:
        out["median_ratio_sub"] = float(np.median(sub))
    return out


def stable_rank(B: torch.Tensor) -> float:
    s = svdvals(B)
    return float((s.sum() ** 2) / (s ** 2).sum())


def write(name: str, payload: dict) -> None:
    import json
    p = RESULTS / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {p.relative_to(REPO)}")


def verdict(name: str, supported: bool, detail: str) -> None:
    print(f"\n=== {name}: {'SUPPORTED' if supported else 'REFUTED'} — {detail}")
