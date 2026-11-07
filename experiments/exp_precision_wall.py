"""
The precision-conditioning wall.

The manuscript asserts, in five places, that polynomial orthogonalizers fail
because "they square the condition number and kappa^2 ~ 1e-7 sits at the edge of
float64".  That sentence conflates two regimes and two precisions.  This script
measures the actual grid.

For every 2-D hidden momentum matrix of the small GPT, at full shape and at
rho = 0.15 row-norm subsampling, and for each working precision, we record:

  * kappa = sigma_min/sigma_max and kappa^2;
  * the orthogonality error of the reference Muon quintic and of the exact Gram
    eigen factor;
  * the descent efficiency eta;
  * whether the computation was even representable (finite, non-NaN).

Predictions this test decides:
  P1. The full-matrix regime is a float64 wall  (kappa^2 ~ 4e-16 ~ eps_fp64).
  P2. The subsampled regime is a float32 wall  (kappa^2 ~ 1e-7 ~ eps_fp32) and
      is comfortable in float64.
  P3. bfloat16, which the published Muon implementation actually uses
      (X = G.bfloat16()), cannot represent either regime.
  P4. In float64 on the subsampled regime the polynomial still fails, so its
      failure there is an iteration-budget (approximation-theoretic) limit, not
      a precision limit.  Increasing K must fix it.

Run:  python exp_precision_wall.py
Out:  ../results/precision_wall.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import data as dm
import robust_orth as ro
from baselines import polar_express
from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}
EPS = {"bf16": 2.0 ** -8, "fp32": 2.0 ** -23, "fp64": 2.0 ** -52}
RHO = 0.15
WARM = 21


# ---------------------------------------------------------------------------
# momentum collection (identical protocol to exp_conditioning.py)
# ---------------------------------------------------------------------------
def collect_momentum():
    torch.manual_seed(0)
    np.random.seed(0)
    dev = torch.device(DEVICE)
    train, val, vocab = dm.prepare_shakespeare()
    gen = torch.Generator().manual_seed(1)
    model = GPT(GPTConfig(vocab_size=vocab, n_layer=6, n_head=6,
                          n_embd=384, block_size=256)).to(dev)
    bufs, grads = {}, {}
    for _ in range(WARM):
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
            bufs[n].mul_(0.95).add_(X, alpha=0.05)
            grads[n] = X.clone()
    return bufs


# ---------------------------------------------------------------------------
# methods, parameterised by working precision
# ---------------------------------------------------------------------------
def muon_ns(X, steps: int, dtype):
    """Verbatim reference routine, but with the working dtype exposed.

    The published implementation is `X = G.bfloat16()`, then
    A = X X^T; B = b A + c A A; X = a X + B X.  No clamp.
    """
    a, b, c = ro.NS_QUINTIC_PUBLISHED
    Z = X.to(dtype)
    Z = Z / (Z.norm() + 1e-7)
    for _ in range(steps):
        A = Z @ Z.mT
        B = b * A + c * (A @ A)
        Z = a * Z + B @ Z
    return Z


def gram_eigh(X, dtype):
    """Exact Gram polar factor in the requested precision (eigh needs fp32+)."""
    Z = X.to(dtype)
    C = Z @ Z.mT
    w, V = torch.linalg.eigh(C)
    w = w.clamp(min=0.0)
    inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
    return (V * inv) @ V.mT @ Z


def exact_svd(X):
    return ro.polar_svd(X, dtype=torch.float64)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@torch.no_grad()
def eta(U, B, B_nuc):
    """Normalised descent efficiency.

    eta(U,B) = <U,B> / (||U||_2 ||B||_*), which is 1 for the exact polar factor
    and <= 1 for every update.  The manuscript's Eq. (7) omits the ||U||_2
    normalisation; without it eta is unbounded (the gram_floor sweep in
    results/gram_floor.json reports eta = 800 for this reason).
    """
    Uf = U.detach().to(torch.float64)
    if not torch.isfinite(Uf).all():
        return float("nan")
    smax = torch.linalg.matrix_norm(Uf, ord=2)
    ip = float((Uf * B).sum())
    return ip / (float(smax) * B_nuc)


@torch.no_grad()
def score(U, B, B_nuc, ref):
    Uf = U.detach().to(torch.float64)
    finite = bool(torch.isfinite(Uf).all())
    out = {"finite": finite,
           "orth_err": ro.orthogonality_error(Uf),
           "eta": eta(Uf, B, B_nuc)}
    if finite:
        s = torch.linalg.svdvals(Uf)
        out["smax"] = float(s[0])
        out["smin"] = float(s[-1])
        out["rel_to_ref"] = float((Uf - ref).norm() / (ref.norm() + 1e-30))
    else:
        out.update(smax=float("nan"), smin=float("nan"), rel_to_ref=float("nan"))
    return out


def main():
    bufs = {k: v.detach().cpu().float() for k, v in collect_momentum().items()}
    torch.save({k: v.clone() for k, v in bufs.items()},
               RESULTS / "momentum_matrices.pt")

    rows = []
    for name, B in bufs.items():
        B64 = B.detach().to(torch.float64)
        B_nuc = float(torch.linalg.svdvals(B64).sum())
        ref_full = exact_svd(B).to(torch.float64)

        idx = ro.select_rows(B, RHO, None, mode="norm")
        sub = B[idx]
        sub64 = sub.detach().to(torch.float64)
        sub_nuc = float(torch.linalg.svdvals(sub64).sum())
        ref_sub = exact_svd(sub).to(torch.float64)

        for regime, X, X64, ref, nuc in (
                ("full", B, B64, ref_full, B_nuc),
                ("sub", sub, sub64, ref_sub, sub_nuc)):
            s = torch.linalg.svdvals(X64)
            kappa = float(s[-1] / s[0])
            sref = torch.linalg.svdvals(ref)
            base = {
                "param": name,
                "regime": regime,
                "shape": list(X.shape),
                "kappa": kappa,
                "kappa2": kappa ** 2,
                # the reference polar factor's own defect sets the floor
                "ref_orth_err": ro.orthogonality_error(ref),
                "ref_smax": float(sref[0]),
                "ref_smin": float(sref[-1]),
            }

            # exact SVD is the ground truth, not a comparison arm
            rows.append({**base, "method": "exact_svd", "dtype": "fp64",
                         "K": None, **score(ref, X64, nuc, ref)})

            for dt in ("fp32", "fp64"):
                try:
                    U = gram_eigh(X, DTYPES[dt])
                    r = score(U, X64, nuc, ref)
                except Exception as e:  # noqa: BLE001
                    r = {"finite": False, "orth_err": float("nan"),
                         "eta": float("nan"), "error": type(e).__name__}
                rows.append({**base, "method": "gram_eigh", "dtype": dt,
                             "K": None, **r})

            ks = (5, 10) if regime == "full" else (5, 10, 20, 40)
            for dt in ("bf16", "fp32", "fp64"):
                for K in ks:
                    try:
                        U = muon_ns(X, K, DTYPES[dt])
                        r = score(U, X64, nuc, ref)
                    except Exception as e:  # noqa: BLE001
                        r = {"finite": False, "orth_err": float("nan"),
                             "eta": float("nan"), "error": type(e).__name__}
                    rows.append({**base, "method": "muon_ns", "dtype": dt,
                                 "K": K, **r})

            pe_steps = (5,) if regime == "full" else (5, 8)
            for dt in ("bf16", "fp32", "fp64"):
                for K in pe_steps:
                    try:
                        U = polar_express(X.to(DTYPES[dt]), steps=K)
                        r = score(U, X64, nuc, ref)
                    except Exception as e:  # noqa: BLE001
                        r = {"finite": False, "orth_err": float("nan"),
                             "eta": float("nan"), "error": type(e).__name__}
                    rows.append({**base, "method": "polar_express", "dtype": dt,
                                 "K": K, **r})

    out = {"warmup_steps": WARM, "rho": RHO, "eps": EPS, "rows": rows}
    (RESULTS / "precision_wall.json").write_text(json.dumps(out, indent=2))

    # ---- summary ----------------------------------------------------------
    print(f"\n{'regime':6s} {'method':14s} {'dt':5s} {'K':>3s} "
          f"{'kappa^2':>10s} {'orth_err':>10s} {'eta':>7s} {'finite':>6s}")
    print("-" * 72)
    seen = set()
    for r in rows:
        key = (r["regime"], r["method"], r["dtype"], r["K"])
        if key in seen:
            continue
        seen.add(key)
        vals = [x for x in rows if (x["regime"], x["method"], x["dtype"],
                                    x["K"]) == key]
        err = np.nanmedian([x["orth_err"] for x in vals])
        e = np.nanmedian([x["eta"] for x in vals])
        fin = sum(1 for x in vals if x["finite"])
        print(f"{r['regime']:6s} {r['method']:14s} {r['dtype']:5s} "
              f"{str(r['K']):>3s} {r['kappa2']:10.2e} {err:10.2e} {e:7.3f} "
              f"{fin:3d}/{len(vals)}")
    print(f"\nwrote {RESULTS / 'precision_wall.json'}")


if __name__ == "__main__":
    main()
