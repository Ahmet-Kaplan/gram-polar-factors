"""
The update-level half of H13.

The training experiment gives identical losses in all four arms, which by itself
is uninformative: it could mean the configurations produce the same update, or it
could mean the loss is insensitive.  This script separates the two by measuring
the update itself on the sixteen real momentum buffers.

For each buffer: the exact factor with a float64 Gram (reference), the exact
factor with a float32 Gram, and the factor truncated at 1e-7 lambda_max with each
precision.  Reported as the relative Frobenius difference from the reference,
median over the sixteen matrices.
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

OUT = ROOT / "results" / "precision_update.json"
THETA = 1e-7


def update(B, gdt, theta):
    G = (B.to(getattr(torch, gdt)) @ B.to(getattr(torch, gdt)).T).to(torch.float64)
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    if theta:
        lam = torch.where(lam < theta * lam[0], torch.zeros_like(lam), lam)
    w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
    U = (V * w) @ (V.T @ B)
    return U / U.pow(2).mean().sqrt().clamp(min=1e-300)


def main() -> None:
    raw, perm = C.corpus()
    model = C.make_model(seed=0)
    B, _ = C.momentum(model, raw, perm, 0, 21, buf_dtype=torch.float64)
    diffs = {"exact_fp32": [], "trunc_fp64": [], "trunc_fp32": []}
    for name, Bm in B.items():
        Bm = Bm.detach().cpu().to(torch.float64)
        if Bm.shape[0] > Bm.shape[1]:
            Bm = Bm.T
        U = {k: update(Bm, g, th) for k, (g, th) in
             {"exact_fp64": ("float64", 0.0), "exact_fp32": ("float32", 0.0),
              "trunc_fp64": ("float64", THETA),
              "trunc_fp32": ("float32", THETA)}.items()}
        # the precision effect on each map, comparing only the precision
        diffs["exact_fp32"].append(float((U["exact_fp32"] - U["exact_fp64"]).norm()
                                        / U["exact_fp64"].norm()))
        diffs["trunc_fp32"].append(float((U["trunc_fp32"] - U["trunc_fp64"]).norm()
                                        / U["trunc_fp64"].norm()))
        # and the cost of the truncation itself, for context
        diffs["trunc_fp64"].append(float((U["trunc_fp64"] - U["exact_fp64"]).norm()
                                        / U["exact_fp64"].norm()))
    out = {"config": {"theta": THETA, "n_matrices": len(diffs["exact_fp32"]),
                      "machine": platform.machine()},
           "relative_update_difference": {}}
    for k, v in diffs.items():
        out["relative_update_difference"][k] = {
            "median": float(np.median(v)), "min": float(np.min(v)),
            "max": float(np.max(v))}
        label = {"exact_fp32": "fp32 effect, exact map",
                 "trunc_fp32": "fp32 effect, truncated map",
                 "trunc_fp64": "cost of the truncation"}[k]
        print(f"  {label:28s} median {np.median(v):.3e} "
              f"[{np.min(v):.2e}, {np.max(v):.2e}]")
    r = out["relative_update_difference"]
    out["verdict"] = {
        "truncation_reduces_precision_sensitivity":
            float(r["exact_fp32"]["median"] / r["trunc_fp32"]["median"]),
        "prediction_holds": bool(r["trunc_fp32"]["median"]
                                 < r["exact_fp32"]["median"] / 5)}
    print(f"  truncation reduces the fp32 effect by "
          f"{out['verdict']['truncation_reduces_precision_sensitivity']:.1f}x")
    Path(OUT).write_text(json.dumps(out, indent=2, default=float))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
