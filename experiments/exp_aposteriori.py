"""
Two things are established here, and the second was not intended.

1. The constant eta in the criterion can be *measured*, not assumed.  Form the
   Gram matrix in the working dtype and again in 80-bit extended precision; the
   discrepancy is the error the theorem's E models.  No unknown constant remains
   and no worst-case gamma_m is invoked.

2. The bound printed in an earlier version of the manuscript was one-sided and
   is violated.  With lambda_hat_i = lambda_i + delta_i and
   |delta_i| <= eta lambda_1, the ratio s_i = (lambda_i/lambda_hat_i)^(1/2) can
   move both ways:

       s_i in [ (1 + t_i)^(-1/2), (1 - t_i)^(-1/2) ],   t_i = eta lambda_1/lambda_i,

   so max_i |s_i - 1| <= (1 - eta kappa^2)^(-1/2) - 1 for eta kappa^2 < 1.  The
   one-sided form keeps only the lower branch, which bounds a *lost* direction,
   and misses the *amplified* one.

3. The two lemmas of Sec. 3.4 are checked here as well:
   * cost: truncating at theta*lambda_hat_max discards exactly the directions
     with computed sigma_hat_i < sqrt(theta) sigma_hat_max, and the retained
     fraction of the nuclear norm is what the descent efficiency loses;
   * benefit: after truncating at theta = eta_hat, every retained direction
     satisfies s_i <= sqrt(2), whereas without truncation s_i is unbounded as
     lambda_hat_i -> 0+.  Measured below as the largest singular value of the
     computed factor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import platform

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "aposteriori.json"

M, N = 192, 576
KAPPAS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
SAFETY = 2.0


def make_matrix(kappa: float, m: int, n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


def _require_extended() -> None:
    """Refuse to run where the "extended" reference is not extended.

    numpy.longdouble is 80-bit on x86-64 but 64-bit on arm64 macOS.  On the
    latter the reference would have the same precision as the quantity it is
    meant to measure, and eta_hat would come out as the difference between two
    float64 matmul kernels -- a number of the right order of magnitude and no
    meaning at all.  This guard exists because that happened.
    """
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise SystemExit(
            "np.longdouble is not wider than float64 on this machine "
            f"({platform.machine()}, eps {np.finfo(np.longdouble).eps:.3e}). "
            "Run this on x86-64, or install a true extended-precision reference.")


def gram_ext(B: torch.Tensor) -> np.ndarray:
    Bl = B.numpy().astype(np.longdouble)
    return Bl @ Bl.T


def polar_svd(B: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return U @ Vh


def cell(B: torch.Tensor, dtype, mode: str = "plain") -> dict:
    """mode: "plain" (clamp negatives), "truncated" (cut below eta_hat), or
    "shifted" (add eta_hat * lambda_max to every eigenvalue)."""
    truncate = mode == "truncated"
    Bd = B.to(dtype)
    G = (Bd @ Bd.T).to(torch.float64)
    Ref = gram_ext(B)
    nr = float(np.linalg.norm(np.asarray(Ref, dtype=np.float64), 2))
    eta_hat = SAFETY * float(np.linalg.norm(G.numpy()
                                           - Ref.astype(np.float64), 2)) / nr

    lam, V = torch.linalg.eigh(G)
    # The eigensolver's own backward error, measured: G V = V diag(lam) up to a
    # residual R, and any matrix F with norm ||R|| reproduces the computed
    # eigenpairs exactly.  So eta_total <= eta_formation + ||R||/||G||.
    resid = float(torch.linalg.matrix_norm(G @ V - V * lam, ord=2)) / \
        float(torch.linalg.matrix_norm(G, ord=2))
    eta_total = eta_hat + resid
    lam_raw = lam
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    lam_max = float(lam[0])
    lam_raw = lam_raw[o]
    n_above_edge = int((lam > eta_total * lam_max).sum())

    if truncate:
        lam_cut = torch.where(lam < eta_total * lam_max,
                              torch.zeros_like(lam), lam)
    elif mode == "shifted":
        lam_cut = lam_raw + eta_total * lam_max
    else:
        lam_cut = lam
    w = torch.where(lam_cut > 0, lam_cut.clamp(min=1e-300).rsqrt(),
                    torch.zeros_like(lam_cut))
    P = (V * w) @ (V.T @ B)
    obs = float(torch.linalg.matrix_norm(P - polar_svd(B), ord=2))

    # kappa of the INPUT, from a float64 SVD.  The ratio of computed Gram
    # eigenvalues is censored at eta^(-1/2), which made eta*kappa^2 = 1 by
    # construction in every void cell of an earlier version.
    s64 = torch.linalg.svdvals(B)
    kappa_true = float(s64[0] / s64[-1])
    kappa_computed = (lam_max / max(float(lam[-1]), eta_total * lam_max)) ** 0.5
    kappa2 = kappa_true ** 2
    koef = eta_total * kappa2
    bound = (1.0 - koef) ** -0.5 - 1.0 if koef < 1.0 else float("inf")
    one_sided = 1.0 - (1.0 + koef) ** -0.5 if koef < 1e6 else 1.0

    sv_P = torch.linalg.svdvals(P)
    # retained fraction of the nuclear norm of the buffer
    sv_B = torch.linalg.svdvals(B.to(dtype).to(torch.float64))
    keep = int((sv_B / float(sv_B[0]) >= np.sqrt(eta_total)).sum())
    nuclear_kept = float(sv_B[:keep].sum() / sv_B.sum())

    return {"eta_hat": eta_hat, "eta_residual": resid,
            "eta_total": eta_total, "kappa": kappa_true,
            "kappa_computed": kappa_computed, "mode": mode,
            "eta_hat_kappa_sq": koef,
            "bound_two_sided": bound, "bound_one_sided": one_sided,
            "observed": obs,
            "two_sided_valid": bool(bound >= obs - 1e-9),
            "one_sided_valid": bool(one_sided >= obs - 1e-9),
            "truncated": bool(truncate),
            "shift_bound": koef,
            "n_directions_above_edge": n_above_edge,
            "n_directions_kept": int((lam_cut > 0).sum()),
            "sigma_max_of_computed_factor": float(sv_P[0]),
            "nuclear_fraction_retained": nuclear_kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    _require_extended()
    print(f"reference precision: longdouble eps "
          f"{np.finfo(np.longdouble).eps:.3e} "
          f"({np.finfo(np.longdouble).bits} bits) on {platform.machine()}")
    rows = []
    print(f"{'kappa':>8} {'dtype':>8} {'eta':>8} {'eta*k^2':>9} "
          f"{'bound':>7} {'obs':>8} {'|P|2':>8} {'keep':>4} {'frac*':>6}")
    for kap in KAPPAS:
        B = make_matrix(kap, M, N)
        for nm, dt in (("float64", torch.float64), ("float32", torch.float32)):
            for mode in ("plain", "truncated", "shifted"):
                r = cell(B, dt, mode=mode)
                r.update({"kappa_target": kap, "gram_dtype": nm})
                rows.append(r)
                lab = nm + {"plain": "", "truncated": "+trunc",
                            "shifted": "+shift"}[mode]
                b = r["bound_two_sided"]
                print(f"{kap:8.0e} {lab:>8} {r['eta_hat']:8.1e} "
                      f"{r['eta_hat_kappa_sq']:9.2e} {b:7.3f} "
                      f"{r['observed']:8.4f} "
                      f"{r['sigma_max_of_computed_factor']:8.3f} "
                      f"{r['n_directions_kept']:4d} "
                      f"{r['nuclear_fraction_retained']:6.3f}")
    plain = [r for r in rows if r["mode"] == "plain"]
    trunc = [r for r in rows if r["mode"] == "truncated"]
    shift = [r for r in rows if r["mode"] == "shifted"]
    summary = {
        "config": {"m": M, "n": N, "kappas": KAPPAS, "safety": SAFETY,
                   "machine": platform.machine(),
                   "longdouble_eps": float(np.finfo(np.longdouble).eps)},
        "rows": rows,
        "two_sided_valid_in": sum(r["two_sided_valid"] for r in plain),
        "one_sided_valid_in": sum(r["one_sided_valid"] for r in plain),
        "n_void": sum(r["eta_hat_kappa_sq"] >= 1 for r in plain),
        "shifted_max_sigma_P": max(r["sigma_max_of_computed_factor"] for r in shift),
        "shifted_within_bound": all(r["observed"] <= r["shift_bound"] * (1 + 1e-6)
                                    for r in shift),
        "shifted_max_observed_error": max(r["observed"] for r in shift),
        "of": len(plain),
        "worst_two_sided_ratio": max(r["bound_two_sided"]
                                     / max(r["observed"], 1e-12) for r in plain
                                     if np.isfinite(r["bound_two_sided"])),
        "max_observed_vs_one_sided": max(r["observed"] - r["bound_one_sided"]
                                         for r in plain),
        # the two lemmas
        "untruncated_max_sigma_P": max(r["sigma_max_of_computed_factor"]
                                       for r in plain),
        "truncated_max_sigma_P": max(r["sigma_max_of_computed_factor"]
                                     for r in trunc),
        "truncated_max_observed_error": max(r["observed"] for r in trunc),
        "untruncated_max_observed_error": max(r["observed"] for r in plain),
        "min_nuclear_fraction_retained": min(r["nuclear_fraction_retained"]
                                             for r in trunc),
    }
    Path(args.out).write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nuntruncated: bound valid {summary['two_sided_valid_in']}/{summary['of']}, "
          f"one-sided {summary['one_sided_valid_in']}/{summary['of']}")
    print(f"largest singular value of the computed factor: "
          f"{summary['untruncated_max_sigma_P']:.2f} untruncated, "
          f"{summary['truncated_max_sigma_P']:.3f} truncated (lemma: sqrt 2 = 1.414)")
    print(f"largest observed error: "
          f"{summary['untruncated_max_observed_error']:.2f} untruncated, "
          f"{summary['truncated_max_observed_error']:.3f} truncated")
    print(f"nuclear fraction retained by the truncation: at least "
          f"{summary['min_nuclear_fraction_retained']:.4f}")
    print(f"wrote {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
