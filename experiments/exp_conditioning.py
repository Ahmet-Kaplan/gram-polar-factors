"""
Collect the spectral-conditioning measurements that drive the paper's argument.

Produces results/conditioning.json with, for every 2-D hidden matrix of the
small GPT:

  * the conditioning ratio sigma_min/sigma_max of the raw gradient and of the
    momentum buffer (EMA, beta = 0.95);
  * the same ratio for row-subsampled submatrices at several rho, under three
    selection rules (row norm, curvature proxy, uniform random);
  * the exact orthogonality error of the polar factor computed by three methods
    on each submatrix, with timings.

This is the evidence for the paper's central claim: subsampling repairs the
spectrum, and that repair is what makes a Gram-based exact polar factor both
accurate and cheap.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import data as dm
import robust_orth as ro
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

RHOS = (0.05, 0.10, 0.15, 0.25, 0.50, 1.00)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def cond_ratio(X: torch.Tensor) -> float:
    s = torch.linalg.svdvals(X.float().cpu())
    if s.numel() == 0 or float(s[0]) == 0:
        return float("nan")
    return float(s[-1] / s[0])


def timed(fn, n=3):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device(DEVICE)
    train, val, vocab = dm.prepare_shakespeare()
    gen = torch.Generator().manual_seed(1)
    model = GPT(GPTConfig(vocab_size=vocab, n_layer=6, n_head=6,
                          n_embd=384, block_size=256)).to(dev)

    # Accumulate momentum buffers exactly as the optimizer would.
    bufs, grads = {}, {}
    WARM = 21
    for step in range(WARM):
        x, y = dm.get_batch(train, 256, 32, dev, gen)
        _, loss = model(x, y)
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is None or p.ndim != 2 or "wte" in n or "wpe" in n:
                continue
            X = (p.grad.T if p.grad.shape[0] > p.grad.shape[1] else p.grad).float()
            if n not in bufs:
                bufs[n] = torch.zeros_like(X)
                grads[n] = X.clone()
            bufs[n].mul_(0.95).add_(X, alpha=0.05)
            grads[n] = X.clone()

    out = {"warmup_steps": WARM, "device": DEVICE, "matrices": []}
    print(f"{'param':30s} {'shape':>12} {'grad':>10} {'EMA':>10} "
          f"{'sub(0.15)':>10} {'k':>5}")
    for n, B in bufs.items():
        G = grads[n]
        m, kk = B.shape
        rec = {
            "name": n,
            "shape": [int(m), int(kk)],
            "grad_ratio": cond_ratio(G),
            "ema_ratio": cond_ratio(B),
            "sub": [],
        }
        # selection-rule comparison at rho = 0.15
        for mode in ("norm", "curvature", "random"):
            w = None
            if mode == "curvature":
                row_sq = (B * B).sum(dim=1)
                w = row_sq * (1.0 + 0.5 * torch.rand_like(row_sq))
            idx = ro.select_rows(B, 0.15, w, mode=mode, seed=0)
            sub = B[idx]
            E_sub = ro.polar_svd(sub)
            rec.setdefault("select_at_015", {})[mode] = {
                "k": int(idx.numel()),
                "sub_ratio": cond_ratio(sub),
                "gram_err": ro.orthogonality_error(ro.polar_gram(sub)),
                # reference Muon quintic: no clamp. The clamped variant is our
                # own modification and is recorded separately (see FINDINGS 11.3)
                "ns_err": ro.orthogonality_error(
                    ro.muon_reference(sub, 5)),
                "ns_clamp_err": ro.orthogonality_error(
                    ro.polar_ns(sub, 5, ro.NS_QUINTIC_PUBLISHED, "frobenius", True)),
                "svd_err": ro.orthogonality_error(E_sub),
            }
        # rho sweep under row-norm selection, with timings
        for rho in RHOS:
            idx = ro.select_rows(B, rho, None, mode="norm")
            sub = B[idx]
            rec["sub"].append({
                "rho": rho,
                "k": int(idx.numel()),
                "sub_ratio": cond_ratio(sub),
                "gram_err": ro.orthogonality_error(ro.polar_gram(sub)),
                "gram_ms": timed(lambda s=sub: ro.polar_gram(s)),
                # reference Muon quintic: no clamp. The clamped variant is our
                # own modification and is recorded separately (see FINDINGS 11.3)
                "ns_err": ro.orthogonality_error(
                    ro.muon_reference(sub, 5)),
                "ns_clamp_err": ro.orthogonality_error(
                    ro.polar_ns(sub, 5, ro.NS_QUINTIC_PUBLISHED, "frobenius", True)),
                "ns_ms": timed(lambda s=sub: ro.polar_ns(
                    s, 5, ro.NS_QUINTIC_PUBLISHED, "frobenius", True)),
            })
        if rho_full := [s for s in rec["sub"] if s["rho"] == 1.00]:
            rec["full"] = {
                "gram_err": rho_full[0]["gram_err"],
                "gram_ms": rho_full[0]["gram_ms"],
                "ns_err": rho_full[0]["ns_err"],
                "svd_err": ro.orthogonality_error(ro.polar_svd(B)),
                "svd_ms": timed(lambda: ro.polar_svd(B)),
            }
        out["matrices"].append(rec)
        s015 = rec["select_at_015"]["norm"]["sub_ratio"]
        print(f"{n.replace('transformer.','')[:30]:30s} "
              f"{str(rec['shape']):>12} {rec['grad_ratio']:10.2e} "
              f"{rec['ema_ratio']:10.2e} {s015:10.2e} "
              f"{rec['select_at_015']['norm']['k']:5d}")

    (RESULTS / "conditioning.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS/'conditioning.json'}")
    gr = np.array([m["grad_ratio"] for m in out["matrices"]])
    er = np.array([m["ema_ratio"] for m in out["matrices"]])
    print(f"grad ratio: median {np.median(gr):.2e}  range [{gr.min():.2e}, {gr.max():.2e}]")
    print(f"EMA  ratio: median {np.median(er):.2e}  range [{er.min():.2e}, {er.max():.2e}]")


if __name__ == "__main__":
    main()
