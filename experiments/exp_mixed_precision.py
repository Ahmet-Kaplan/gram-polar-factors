"""
Which half of the pipeline should get the extra precision?

Eq. (etatotal) of the paper splits the error into a Gram-formation term and an
eigensolver term.  The split predicts, for each mixed-precision configuration,
which term dominates -- and therefore whether paying for a float64
eigensolver buys anything once the Gram matrix has been formed in float32.

Four configurations, over a grid of condition numbers:
  (fp64 Gram, fp64 eigh)   the reference
  (fp32 Gram, fp64 eigh)   what implementations actually do
  (fp64 Gram, fp32 eigh)   never used, but it isolates the eigensolver term
  (fp32 Gram, fp32 eigh)   fully reduced

For each: eta_formation (against an 80-bit reference), eta_eigh (the residual
||Ghat V - V Lambda||/||Ghat|| measured in the eigensolver's own dtype),
eta_total, the criterion eta_total*kappa^2, the bound, and the observed error
against a float64 SVD polar factor.

Requires an x86-64 machine: numpy.longdouble is 64-bit on arm64 macOS, and the
guard below refuses to run there.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "mixed_precision.json"
M, N = 192, 576
KAPPAS = [1e-2, 1e-4, 1e-6, 1e-8]
CONFIGS = [("float64", "float64"), ("float32", "float64"),
           ("float64", "float32"), ("float32", "float32")]


def _require_extended() -> None:
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise SystemExit(f"longdouble is not wider than float64 on "
                         f"{platform.machine()}; run on x86-64")


def make_matrix(kappa, m=M, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


def gram_ext(B):
    Bl = B.detach().cpu().numpy().astype(np.longdouble)
    return Bl @ Bl.T


def cell(B, gdt_name, edt_name) -> dict:
    gdt, edt = getattr(torch, gdt_name), getattr(torch, edt_name)
    G64 = B @ B.T
    G = (B.to(gdt) @ B.to(gdt).T).to(torch.float64)
    Ref = gram_ext(B)
    nr = float(np.linalg.norm(np.asarray(Ref, dtype=np.float64), 2))
    eta_form = 2.0 * float(np.linalg.norm(G.numpy() - Ref.astype(np.float64), 2)) / nr

    Ge = G.to(edt)                      # the eigensolver sees this
    lam, V = torch.linalg.eigh(Ge)
    lam64 = lam.to(torch.float64).clamp(min=0.0)
    V64 = V.to(torch.float64)
    # the residual of the computed eigenpairs, measured against Ge in float64
    res = float(torch.linalg.matrix_norm(Ge.to(torch.float64) @ V64 - V64 * lam64,
                                         ord=2)) / \
        float(torch.linalg.matrix_norm(Ge.to(torch.float64), ord=2))
    o = torch.argsort(lam64, descending=True)
    lam64, V64 = lam64[o], V64[:, o]
    w = torch.where(lam64 > 0, lam64.rsqrt(), torch.zeros_like(lam64))
    P = (V64 * w) @ (V64.T @ B)

    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    obs = float(torch.linalg.matrix_norm(P - U @ Vh, ord=2))
    lam_max, lam_min = float(lam64[0]), float(lam64[-1])
    eta_tot = eta_form + res
    lam_eff = max(lam_min, eta_tot * lam_max)
    k2 = lam_max / max(lam_eff, 1e-300)
    koef = eta_tot * k2
    bound = (1.0 - koef) ** -0.5 - 1.0 if koef < 1.0 else float("inf")
    return {"kappa_target": None, "gram": gdt_name, "eigh": edt_name,
            "eta_formation": eta_form, "eta_eigh": res, "eta_total": eta_tot,
            "kappa": k2 ** 0.5, "eta_total_kappa_sq": koef,
            "bound": bound, "observed": obs,
            "dominant_term": "formation" if eta_form > res else "eigensolver"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    _require_extended()
    print(f"reference: longdouble eps {np.finfo(np.longdouble).eps:.3e} "
          f"on {platform.machine()}")
    rows = []
    print(f"{'kappa':>8} {'gram':>8} {'eigh':>8} {'eta_form':>10} {'eta_eigh':>10} "
          f"{'dominant':>11} {'eta*k^2':>9} {'observed':>10}")
    for kap in KAPPAS:
        B = make_matrix(kap)
        for gdt, edt in CONFIGS:
            r = cell(B, gdt, edt)
            r["kappa_target"] = kap
            rows.append(r)
            print(f"{kap:8.0e} {gdt:>8} {edt:>8} {r['eta_formation']:10.2e} "
                  f"{r['eta_eigh']:10.2e} {r['dominant_term']:>11} "
                  f"{r['eta_total_kappa_sq']:9.2e} {r['observed']:10.4f}")

    def get(g, e, kap):
        return next(r for r in rows if r["gram"] == g and r["eigh"] == e
                    and r["kappa_target"] == kap)

    # the claim the split makes: with an fp32 Gram the eigensolver is irrelevant
    f32g = [get("float32", "float64", k)["observed"] for k in KAPPAS]
    f32b = [get("float32", "float32", k)["observed"] for k in KAPPAS]
    both = [x / y if y > 0 else float("inf") for x, y in zip(f32g, f32b)]
    # and with an fp64 Gram the eigensolver dominates in float64 but not in fp32
    d1 = get("float64", "float64", 1e-6)
    d2 = get("float64", "float32", 1e-6)
    out = {"config": {"m": M, "n": N, "kappas": KAPPAS,
                      "machine": platform.machine(),
                      "longdouble_eps": float(np.finfo(np.longdouble).eps)},
           "rows": rows,
           "fp64_eigh_vs_fp32_eigh_on_fp32_gram": both,
           "fp64_gram_isolates_eigensolver": {
               "fp64_eigh": {"eta_eigh": d1["eta_eigh"],
                             "observed": d1["observed"]},
               "fp32_eigh": {"eta_eigh": d2["eta_eigh"],
                             "observed": d2["observed"]}},
           "eigensolver_irrelevant_when_gram_is_fp32": bool(
               all(abs(x - 1.0) < 0.2 for x in both))}
    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nobserved error, fp32 Gram: fp64 eigh / fp32 eigh over the grid = "
          f"{[round(x, 3) for x in both]}")
    print(f"  => a float64 eigensolver buys nothing once the Gram is fp32: "
          f"{out['eigensolver_irrelevant_when_gram_is_fp32']}")
    print(f"at kappa=1e-6, fp64 Gram: eta_eigh {d1['eta_eigh']:.2e} (fp64) vs "
          f"{d2['eta_eigh']:.2e} (fp32); observed {d1['observed']:.3e} vs "
          f"{d2['observed']:.3e}")
    print(f"wrote {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
