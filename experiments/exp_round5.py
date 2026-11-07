"""
Round-5 (5-referee panel) checks and experiments for the NLAA manuscript.

  A. Theorem (general perturbation).  For any symmetric E with ||E||_2 <= eta*lambda_1
     and t = eta*kappa^2 < 1,
         ||P_hat - P||_2      <= (1 - t)^(-1/2) - 1,
         ||P_hat P_hat^T - I|| <= t / (1 - t),
     with P_hat = (G + E)^(-1/2) B.  No assumption on the eigenvectors.  Checked
     against extremal, random, off-diagonal and searched perturbations, and the
     first-order form P_hat - P = D P + O(t^2), D_ij = -E_ij / (s_i (s_i + s_j))
     (in the eigenbasis of G), whose skew part is what re-projection keeps.
  B. Shifted route.  With delta >= eta*lambda_1, P_delta = (G_hat + delta I)^(-1/2) B
     is a contraction for every input and ||P_delta - P|| <= (eta + delta/lambda_1) kappa^2
     / (1 + sqrt(1 - ...)) <= eta kappa^2 when delta = eta*lambda_1.
  C. The invariance "lower bound" of the previous Prop. 3 is false: counterexample.
  D. kappa sweep through the float32 threshold: plain, truncated, shifted.
  E. Newton-Schulz run to convergence in float64 / float32 / bfloat16: precision floors.
  F. Mixed precision on a wider admissible grid, three matrices per cell.
  G. Fitted exponent of the Gram/SVD error ratio of Sec. 3.8.
  H. Self-check that the extended reference is extended (exact rational Gram).

Parts D, F and H need an 80-bit long double (x86-64; Rosetta 2 on Apple silicon works).
"""
from __future__ import annotations

import json
import platform
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "round5.json"
U64, U32 = 2.0 ** -53, 2.0 ** -24   # unit roundoffs (round to nearest)
FLOOR = 100 * U64   # checks of exact identities run in float64: allow 100 u kappa^2


def make_matrix(kappa: float, m: int, n: int, seed: int = 0) -> torch.Tensor:
    """Prescribed geometric spectrum from 1 down to 1/kappa (kappa >= 1)."""
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, -np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


def polar(B: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(B, full_matrices=False)
    return U @ Vh


def inv_sqrt_apply(A: torch.Tensor, B: torch.Tensor, cut: float = 0.0) -> torch.Tensor:
    lam, V = torch.linalg.eigh(A)
    w = torch.where(lam > cut, lam.clamp(min=1e-300).rsqrt(), torch.zeros_like(lam))
    return (V * w) @ (V.T @ B)


def bound(t: float) -> float:
    return (1.0 - t) ** -0.5 - 1.0 if t < 1.0 else float("inf")


def opn(A: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(A, ord=2))


# --------------------------------------------------------------------- A
def part_a() -> dict:
    m, n = 40, 120
    rng = np.random.default_rng(0)
    rows, worst = [], 0.0
    for kappa in (10.0, 30.0, 100.0, 300.0, 1000.0):
        B = make_matrix(kappa, m, n, seed=1)
        U, s, _ = torch.linalg.svd(B, full_matrices=False)
        G = B @ B.T
        lam1 = float(s[0] ** 2)
        P = polar(B)
        for t in (0.01, 0.1, 0.5, 0.9):
            eta = t / kappa ** 2
            nrm = eta * lam1
            perts = {}
            # extremal: negative shift of the smallest eigenvalue
            e = U[:, -1:]
            perts["extremal"] = -nrm * (e @ e.T)
            # purely off-diagonal coupling of the top and bottom directions
            a, b = U[:, :1], U[:, -1:]
            perts["offdiag"] = nrm * (a @ b.T + b @ a.T)
            for k in range(200):     # random symmetric, and random tail-weighted
                R = torch.from_numpy(rng.standard_normal((m, m)))
                R = R + R.T
                if k % 2:
                    W = U @ torch.diag(torch.linspace(0.0, 1.0, m, dtype=torch.float64) ** 4) @ U.T
                    R = W @ R @ W + R * 1e-3
                R = R / opn(R) * nrm
                perts[f"rand{k}"] = R if k % 3 else -R
            res = {}
            for name, E in perts.items():
                Ph = inv_sqrt_apply(G + E, B)
                err = opn(Ph - P)
                defect = opn(Ph @ Ph.T - torch.eye(m, dtype=torch.float64))
                res[name] = (err, defect)
            emax = max(v[0] for v in res.values())
            dmax = max(v[1] for v in res.values())
            b_err, b_def = bound(t), t / (1.0 - t)
            worst = max(worst, emax / b_err)
            rows.append({"kappa": kappa, "t": t, "bound": b_err,
                         "max_error": emax, "extremal_error": res["extremal"][0],
                         "offdiag_error": res["offdiag"][0],
                         "defect_bound": b_def, "max_defect": dmax,
                         "extremal_defect": res["extremal"][1],
                         # tolerance: the float64 floor of computing P_hat itself
                         "floor": FLOOR * kappa ** 2,
                         "valid": bool(emax <= b_err + FLOOR * kappa ** 2
                                       and dmax <= b_def + FLOOR * kappa ** 2)})
    # first-order form and the skew/symmetric split (small t)
    fo = []
    for kappa in (10.0, 100.0, 1000.0):
        B = make_matrix(kappa, m, n, seed=2)
        U, s, _ = torch.linalg.svd(B, full_matrices=False)
        G, P = B @ B.T, polar(B)
        t = 1e-4
        nrm = t / kappa ** 2 * float(s[0] ** 2)
        R = torch.from_numpy(rng.standard_normal((m, m)))
        R = (R + R.T)
        R = R / opn(R) * nrm
        Ph = inv_sqrt_apply(G + R, B)
        Et = U.T @ R @ U                           # E in the eigenbasis of G
        D = -Et / (s[:, None] * (s[:, None] + s[None, :]))
        Dfull = U @ D @ U.T
        Om = 0.5 * (Dfull - Dfull.T)
        Pp = polar(Ph)
        fo.append({"kappa": kappa,
                   "first_order_rel_residual": opn(Ph - P - Dfull @ P) / opn(Ph - P),
                   "reprojection_rel_residual": opn(Pp - P - Om @ P) / opn(Pp - P),
                   "skew_fraction": opn(Om) / opn(Dfull),
                   "D_norm_over_t_half": opn(Dfull) / (t / 2)})
    return {"rows": rows, "all_valid": all(r["valid"] for r in rows),
            "worst_ratio_to_bound": worst,
            "extremal_attains": all(abs(r["extremal_error"] / r["bound"] - 1) < 1e-6
                                    for r in rows),
            "first_order": fo}


# --------------------------------------------------------------------- B
def part_b() -> dict:
    m, n = 40, 120
    rng = np.random.default_rng(1)
    rows = []
    for kappa in (10.0, 100.0, 1000.0):
        B = make_matrix(kappa, m, n, seed=3)
        U, s, _ = torch.linalg.svd(B, full_matrices=False)
        G, P = B @ B.T, polar(B)
        lam1 = float(s[0] ** 2)
        for t in (0.01, 0.5, 2.0, 50.0):            # includes the void regime
            eta = t / kappa ** 2
            delta = eta * lam1
            emax, smax = 0.0, 0.0
            for k in range(100):
                R = torch.from_numpy(rng.standard_normal((m, m)))
                R = R + R.T
                R = R / opn(R) * eta * lam1
                if k == 0:
                    e = U[:, -1:]
                    R = -eta * lam1 * (e @ e.T)
                Ph = inv_sqrt_apply(G + R + delta * torch.eye(m, dtype=torch.float64), B)
                emax = max(emax, opn(Ph - P))
                smax = max(smax, float(torch.linalg.svdvals(Ph)[0]))
            rows.append({"kappa": kappa, "t": t, "max_error": emax,
                         "shift_bound": t, "max_sigma": smax,
                         "contraction": bool(smax <= 1 + FLOOR * kappa ** 2),
                         "within_bound": bool(emax <= t * (1 + 1e-9))})
    return {"rows": rows,
            "always_contraction": all(r["contraction"] for r in rows),
            "bound_valid": all(r["within_bound"] for r in rows)}


# --------------------------------------------------------------------- C
def part_c() -> dict:
    c = lambda x: (3 * x - x ** 3) / 2
    e = 0.15
    s = np.linspace(e, 1 - 1e-9, 200001)
    g = c(c(s))
    lam = float(((1 - g) / (1 - s)).max())
    return {"eps": e, "lambda_g": lam, "claimed_lower_bound": 1 - e,
            "maps_into_itself": bool(g.min() >= e and g.max() <= 1 + 1e-12),
            "proposition_false": lam < 1 - e}


# --------------------------------------------------------------------- shared
def gram_ext(B: torch.Tensor) -> np.ndarray:
    Bl = B.numpy().astype(np.longdouble)
    return Bl @ Bl.T


def eta_measured(B: torch.Tensor, gdt: torch.dtype, edt: torch.dtype):
    """Formation term against the 80-bit reference, plus the eigensolver residual."""
    Bd = B.to(gdt)
    G = (Bd @ Bd.T).to(torch.float64)
    Ref = gram_ext(B)
    nr = float(np.linalg.norm(Ref.astype(np.float64), 2))
    form = 2.0 * float(np.linalg.norm(G.numpy() - Ref.astype(np.float64), 2)) / nr
    Ge = G.to(edt)
    lam, V = torch.linalg.eigh(Ge)
    lam64, V64 = lam.to(torch.float64), V.to(torch.float64)
    Gd = Ge.to(torch.float64)
    res = opn(Gd @ V64 - V64 * lam64) / opn(Gd)
    return form, res, lam64, V64


def factors(B, lam, V, eta):
    lam_max = float(lam.max())
    out = {}
    for name in ("plain", "truncated", "shifted"):
        if name == "plain":
            l = lam
            w = torch.where(l > 0, l.clamp(min=1e-300).rsqrt(), torch.zeros_like(l))
        elif name == "truncated":
            l = torch.where(lam < eta * lam_max, torch.zeros_like(lam), lam)
            w = torch.where(l > 0, l.clamp(min=1e-300).rsqrt(), torch.zeros_like(l))
        else:
            l = lam + eta * lam_max
            w = l.clamp(min=1e-300).rsqrt()
        Ph = (V * w) @ (V.T @ B)
        out[name] = Ph
    return out


# --------------------------------------------------------------------- D
def part_d() -> dict:
    m, n = 192, 576
    rows = []
    for kappa in np.logspace(1.5, 5.5, 17):
        B = make_matrix(float(kappa), m, n, seed=0)
        P = polar(B)
        form, res, lam, V = eta_measured(B, torch.float32, torch.float64)
        eta = form + res
        F = factors(B, lam, V, eta)
        r = {"kappa": float(kappa), "eta": eta, "t": eta * kappa ** 2,
             "kappa_over_threshold": float(kappa) * U32 ** 0.5,
             "bound": bound(eta * kappa ** 2)}
        for k, Ph in F.items():
            r[f"err_{k}"] = opn(Ph - P)
            r[f"sigma_{k}"] = float(torch.linalg.svdvals(Ph)[0])
            r[f"fro_{k}"] = float(torch.linalg.matrix_norm(Ph - P))
        rows.append(r)
    adm = [r for r in rows if r["t"] < 1]
    return {"m": m, "n": n, "gram": "float32", "eigh": "float64", "rows": rows,
            "bound_valid_admissible": all(r["err_plain"] <= r["bound"] for r in adm),
            "shift_always_contraction": all(r["sigma_shifted"] <= 1 + 1e-6 for r in rows),
            "shift_within_t": all(r["err_shifted"] <= r["t"] * (1 + 1e-6) for r in rows)}


# --------------------------------------------------------------------- E
def part_e() -> dict:
    m, n, iters = 58, 384, 80
    out = {}
    for kappa in (1e2, 1e3):
        B = make_matrix(kappa, m, n, seed=4)
        P = polar(B)
        rec = {}
        for name, dt in (("float64", torch.float64), ("float32", torch.float32),
                         ("bfloat16", torch.bfloat16)):
            X = (B / torch.linalg.matrix_norm(B)).to(dt)
            errs = []
            for _ in range(iters):
                X = 1.5 * X - 0.5 * (X @ X.T) @ X
                errs.append(opn(X.to(torch.float64) - P))
            rec[name] = {"floor": float(min(errs[-20:])), "final": errs[-1],
                         "iters_to_1e-1": next((i + 1 for i, e in enumerate(errs)
                                                if e < 1e-1), None),
                         "error_at_5": errs[4], "error_at_10": errs[9]}
        out[f"{kappa:.0e}"] = rec
    return {"m": m, "n": n, "iterations": iters, "method": "Newton-Schulz cubic",
            "cells": out}


# --------------------------------------------------------------------- F
def part_f() -> dict:
    m, n = 192, 576
    configs = [("float64", "float64"), ("float32", "float64"), ("float64", "float32"),
               ("float32", "float32"), ("float16", "float32"), ("bfloat16", "float32")]
    kappas = [1e1, 3e1, 1e2, 3e2, 1e3]
    out = {}
    for g, e in configs:
        cell = {}
        for kappa in kappas:
            obs, etas = [], []
            for seed in (0, 1, 2):
                B = make_matrix(kappa, m, n, seed=seed)
                P = polar(B)
                form, res, lam, V = eta_measured(B, getattr(torch, g), getattr(torch, e))
                lam = lam.clamp(min=0.0)
                w = torch.where(lam > 0, lam.clamp(min=1e-300).rsqrt(), torch.zeros_like(lam))
                obs.append(opn((V * w) @ (V.T @ B) - P))
                etas.append(form + res)
            et = float(np.median(etas))
            cell[f"{kappa:.0e}"] = {"median": float(np.median(obs)), "min": min(obs),
                                    "max": max(obs), "eta_total": et,
                                    "t": et * kappa ** 2}
        out[f"{g}|{e}"] = cell
    return {"m": m, "n": n, "kappas": kappas, "cells": out}


# --------------------------------------------------------------------- G
def part_g() -> dict:
    th = json.loads((ROOT / "results" / "theory_upgrades.json").read_text())
    rows = th["conditioning"]["rows"]
    k = np.array([r["kappa"] for r in rows], dtype=float)
    q = np.array([r["ratio"] for r in rows], dtype=float)
    slope = float(np.polyfit(np.log10(k), np.log10(q), 1)[0])
    return {"kappa": k.tolist(), "ratio": q.tolist(), "fitted_exponent": slope}


# --------------------------------------------------------------------- H
def part_h() -> dict:
    rng = np.random.default_rng(5)
    A = rng.standard_normal((12, 30))
    exact = [[sum(Fraction(A[i, k]) * Fraction(A[j, k]) for k in range(30))
              for j in range(12)] for i in range(12)]
    L = A.astype(np.longdouble)
    G = L @ L.T
    err = max(abs(Fraction(float(G[i, j])) + Fraction(float(G[i, j] - np.longdouble(float(G[i, j]))))
                  - exact[i][j]) / max(abs(exact[i][j]), Fraction(1, 10 ** 30))
              for i in range(12) for j in range(12))
    return {"longdouble_eps": float(np.finfo(np.longdouble).eps),
            "max_rel_error_vs_exact": float(err)}


def main() -> None:
    torch.set_num_threads(4)
    extended = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps
    out = {"config": {"machine": platform.machine(), "torch": torch.__version__,
                      "longdouble_eps": float(np.finfo(np.longdouble).eps)}}
    for name, fn, needs_ext in (("A_general_theorem", part_a, False),
                                ("B_shift", part_b, False),
                                ("C_prop_bound_counterexample", part_c, False),
                                ("G_power_of_kappa_fit", part_g, False),
                                ("E_newton_schulz_floors", part_e, False),
                                ("H_reference_check", part_h, True),
                                ("D_kappa_sweep_fp32", part_d, True),
                                ("F_mixed_wide", part_f, True)):
        if needs_ext and not extended:
            print(f"skip {name}: long double is not extended on {platform.machine()}")
            continue
        out[name] = fn()
        print(f"done {name}")
    OUT.write_text(json.dumps(out, indent=2, default=float))
    a, b, c = out["A_general_theorem"], out["B_shift"], out["C_prop_bound_counterexample"]
    print(f"A: valid {a['all_valid']}, worst ratio {a['worst_ratio_to_bound']:.6f}, "
          f"extremal attains {a['extremal_attains']}")
    for r in a["first_order"]:
        print(f"   kappa {r['kappa']:.0e}: first-order resid {r['first_order_rel_residual']:.1e}, "
              f"reproj resid {r['reprojection_rel_residual']:.1e}, skew frac {r['skew_fraction']:.3f}, "
              f"||D||/(t/2) {r['D_norm_over_t_half']:.3f}")
    print(f"B: contraction {b['always_contraction']}, bound {b['bound_valid']}")
    print(f"C: lambda(g) {c['lambda_g']:.4f} < {c['claimed_lower_bound']}: {c['proposition_false']}")
    print(f"G: fitted exponent {out['G_power_of_kappa_fit']['fitted_exponent']:.3f}")
    for k, v in out["E_newton_schulz_floors"]["cells"].items():
        print(f"E: kappa {k}: " + ", ".join(f"{d} floor {x['floor']:.1e}" for d, x in v.items()))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
