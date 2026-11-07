"""
Is the Gram route for the polar factor the analogue of CholeskyQR, and does the
CholeskyQR2 fix transfer?

The CholeskyQR literature says: computing the orthogonal factor through the Gram
matrix makes the error quadratic in the condition number, the algorithm is not
backward stable, and a second pass through the same algorithm (CholeskyQR2)
restores first-order behaviour.  This script asks whether the same holds for the
polar factor, because if it does the paper has a mature theory to place itself in
and a new algorithm to propose.

Measurement, on matrices with a prescribed condition number:
  pass 1:  P1 = gram_polar(B)
  pass 2:  P2 = gram_polar(P1)                (the CholeskyQR2 analogue)
  pass 3, 4: the same again, to see whether it stops
and the mechanism: write M = P P1^T (m x m, close to I) and split
M - I into its symmetric and skew parts.  The first pass's error is a *rescaling*
of the singular values, which is the symmetric component; a projection back onto
the isometries removes exactly that.  So the prediction is that pass 2 removes
the symmetric part and leaves the skew part, i.e. the improvement should be
large but not complete.

Verdict:
  SUPPORTED if the error stops depending on kappa after the second pass, i.e. the
  ratio error(pass 2, large kappa) / error(pass 2, small kappa) is O(1) where the
  first pass's ratio grows like kappa^2.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "two_pass.json"

M, N = 192, 576
# kappa chosen so that eta*kappa^2 sweeps from negligible to order one for each dtype
GRID = {"float64": [1e2, 1e3, 1e4, 1e5, 1e6, 3e6],
        "float32": [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]}


def make_matrix(kappa: float, m: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


def gram_polar(X: torch.Tensor, dtype, truncate_at: float | None) -> torch.Tensor:
    """One pass of the Gram route, arithmetic in `dtype`, cut at truncate_at."""
    Xd = X.to(dtype)
    G = (Xd @ Xd.T).to(torch.float64)
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    if truncate_at is not None:
        lam = torch.where(lam < truncate_at * lam[0], torch.zeros_like(lam), lam)
    w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
    return (V * w) @ (V.T @ Xd.to(torch.float64))


def split_error(P: torch.Tensor, Phat: torch.Tensor) -> tuple[float, float, float]:
    """Symmetric (rescaling) and skew (rotation) components of the error."""
    m = P.shape[0]
    Mk = P @ Phat.T
    I = torch.eye(m, dtype=Mk.dtype, device=Mk.device)
    sym = 0.5 * ((Mk - I) + (Mk - I).T)
    skew = 0.5 * ((Mk - I) - (Mk - I).T)
    total = float(torch.linalg.matrix_norm(Phat - P, ord=2))
    return total, float(torch.linalg.matrix_norm(sym, ord=2)), \
        float(torch.linalg.matrix_norm(skew, ord=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    print(f"{platform.machine()}, torch {torch.__version__}")
    rows = []
    for dname, kappas in GRID.items():
        dt = getattr(torch, dname)
        eps = torch.finfo(dt).eps
        print(f"\n=== {dname} (eps {eps:.2e})")
        print(f"{'kappa':>9} {'eta*k^2':>9} {'err p1':>10} {'err p2':>10} "
              f"{'err p3':>10} {'p2/p1':>7} {'sym p1':>9} {'skew p1':>9}")
        for kap in kappas:
            B = make_matrix(kap, M, N)
            U, _, Vh = torch.linalg.svd(B, full_matrices=False)
            Pref = U @ Vh
            # eta for this dtype, from the Gram formation
            G64 = B @ B.T
            Gd = (B.to(dt) @ B.to(dt).T).to(torch.float64)
            eta = 2.0 * float(torch.linalg.matrix_norm(Gd - G64, ord=2)) / \
                float(torch.linalg.matrix_norm(G64, ord=2))
            P1 = gram_polar(B, dt, None)
            P2 = gram_polar(P1, dt, None)
            P3 = gram_polar(P2, dt, None)
            e1, e2, e3 = (float(torch.linalg.matrix_norm(Pk - Pref, ord=2))
                          for Pk in (P1, P2, P3))
            _, sym1, skew1 = split_error(Pref, P1)
            _, sym2, skew2 = split_error(Pref, P2)
            rows.append({"dtype": dname, "kappa": kap, "eta": eta,
                         "eta_kappa_sq": eta * kap ** 2,
                         "err_pass1": e1, "err_pass2": e2, "err_pass3": e3,
                         "ratio_p2_p1": e2 / max(e1, 1e-300),
                         "sym_pass1": sym1, "skew_pass1": skew1,
                         "sym_pass2": sym2, "skew_pass2": skew2})
            print(f"{kap:9.0e} {eta*kap**2:9.2e} {e1:10.3e} {e2:10.3e} "
                  f"{e3:10.3e} {e2/max(e1,1e-300):7.2f} {sym1:9.3e} {skew1:9.3e}")

    def summarize(dname):
        R = [r for r in rows if r["dtype"] == dname]
        small, large = R[0], R[-1]
        return {"err_pass1_first": small["err_pass1"],
                "err_pass1_last": large["err_pass1"],
                "err_pass2_first": small["err_pass2"],
                "err_pass2_last": large["err_pass2"],
                "growth_pass1": large["err_pass1"] / max(small["err_pass1"], 1e-300),
                "growth_pass2": large["err_pass2"] / max(small["err_pass2"], 1e-300),
                "sym_removed_last": 1.0 - large["sym_pass2"] / max(large["sym_pass1"], 1e-300),
                "skew_kept_last": large["skew_pass2"] / max(large["skew_pass1"], 1e-300)}

    out = {"config": {"m": M, "n": N, "grid": GRID, "machine": platform.machine()},
           "rows": rows,
           "summary": {d: summarize(d) for d in GRID}}
    for d, s in out["summary"].items():
        print(f"\n--- {d}: pass 1 grows by {s['growth_pass1']:.1f}x across the grid, "
              f"pass 2 by {s['growth_pass2']:.1f}x")
        print(f"    at the largest kappa: symmetric component removed "
              f"{s['sym_removed_last']:.1%}, skew kept {s['skew_kept_last']:.2f}x")
    supported = all(out["summary"][d]["growth_pass2"]
                    < 0.2 * out["summary"][d]["growth_pass1"] for d in GRID)
    out["verdict"] = {"supported": bool(supported),
                      "growth": {d: (out["summary"][d]["growth_pass1"],
                                     out["summary"][d]["growth_pass2"])
                                 for d in GRID}}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\n=== H12 two-pass polar factor: "
          f"{'SUPPORTED' if supported else 'REFUTED'}")
    print(f"wrote {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
