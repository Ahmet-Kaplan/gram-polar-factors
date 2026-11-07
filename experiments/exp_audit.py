"""
Audit of the shipped implementation, against the paper's own tables.

Muon's reference implementation (KellerJordan/Muon, muon.py) runs

    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)     # Frobenius, not spectral
    for _ in range(steps): A = X @ X.mT; X = a*X + (b*A + c*A@A) @ X

with steps = 5 and bfloat16 hard-coded.  Its docstring says the iteration does
not converge, that the singular values come out roughly Uniform(0.5, 1.5), and
that this "turns out not to hurt model performance at all".

This script asks the quantitative question the docstring leaves open: at five
steps, and at more steps, what does each dtype buy, on the real momentum buffers?
The answer uses the paper's own rule --- the dtype sets a floor that additional
iterations cannot pass.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "novelty"))
import common as C  # noqa: E402

OUT = ROOT / "results" / "audit.json"
STEPS = [5, 10, 20, 40, 80]
DTYPES = ["float64", "float32", "bfloat16"]
A, B_, C_ = 3.4445, -4.7750, 2.0315


def zeropower(G: torch.Tensor, steps: int, dt) -> torch.Tensor:
    """Muon's own routine, with the dtype made a parameter."""
    X = G.to(dt)
    if X.shape[-2] > X.shape[-1]:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        Aa = X @ X.mT
        Bb = B_ * Aa + C_ * Aa @ Aa
        X = A * X + Bb @ X
    if X.shape[-2] > X.shape[-1]:
        X = X.mT
    return X.float().double()


def orth_err(P: torch.Tensor) -> float:
    m = P.shape[0]
    I = torch.eye(m, dtype=P.dtype)
    return float((P @ P.T - I).norm() / m ** 0.5)


def main() -> None:
    print(f"{platform.machine()}")
    raw, perm = C.corpus()
    model = C.make_model(seed=0)
    B, _ = C.momentum(model, raw, perm, 0, 21, buf_dtype=torch.float64)
    rows = {}
    for dt in DTYPES:
        for k in STEPS:
            errs, smins, norms = [], [], []
            for name, Bm in B.items():
                Bm = Bm.detach().cpu().to(torch.float64)
                if Bm.shape[0] > Bm.shape[1]:
                    Bm = Bm.T
                P = zeropower(Bm, k, getattr(torch, dt))
                errs.append(orth_err(P))
                s = torch.linalg.svdvals(P)
                smins.append(float(s[-1]))
                norms.append(float(s[0]))
            rows[f"{dt}|{k}"] = {"median_orth_err": float(np.median(errs)),
                                 "median_sigma_min": float(np.median(smins)),
                                 "median_sigma_max": float(np.median(norms))}
            r = rows[f"{dt}|{k}"]
            print(f"  {dt:9s} steps {k:3d}  orth err {r['median_orth_err']:.4f}  "
                  f"sigma in [{r['median_sigma_min']:.4f}, {r['median_sigma_max']:.4f}]")
    floor = {dt: min(rows[f"{dt}|{k}"]["median_orth_err"] for k in STEPS) for dt in DTYPES}
    five = {dt: rows[f"{dt}|5"]["median_orth_err"] for dt in DTYPES}
    out = {"config": {"steps": STEPS, "dtypes": DTYPES,
                      "machine": platform.machine()},
           "rows": rows, "floor": floor, "at_five_steps": five,
           "gain_from_more_steps": {dt: five[dt] - floor[dt] for dt in DTYPES}}
    print(f"\n  orthogonality error at 5 steps: "
          + ", ".join(f"{dt} {five[dt]:.3f}" for dt in DTYPES))
    print(f"  floor reachable at 80 steps:    "
          + ", ".join(f"{dt} {floor[dt]:.3f}" for dt in DTYPES))
    print(f"  gain from iterating further:    "
          + ", ".join(f"{dt} {out['gain_from_more_steps'][dt]:.3f}" for dt in DTYPES))
    out["verdict"] = {
        "precision_irrelevant_at_five_steps":
            bool(max(five.values()) - min(five.values()) < 0.05),
        "bf16_floor_far_above_float64": bool(floor["bfloat16"] > 10 * max(floor["float64"], 1e-12)),
        "iterating_helps_in_float64_not_in_bf16":
            bool(out["gain_from_more_steps"]["float64"] > 0.3
                 and out["gain_from_more_steps"]["bfloat16"] < 0.2),
    }
    print(f"\n  verdict: {out['verdict']}")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
