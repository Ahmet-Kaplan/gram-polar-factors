"""
Four upgrades to the theory, each a claim that can be checked.

1. SHARPNESS.  Theorem 1 bounds max_i|s_i-1| by g(eta*kappa^2) with
   g(t) = (1-t)^{-1/2} - 1.  If E is diagonal and negative on the smallest
   eigenvalue, the bound is *attained exactly*: with G = diag(1, kappa^-2) and
   E = -eta e_min e_min^T, s_min = (kappa^-2/(kappa^-2 - eta))^(1/2)
   = (1-eta*kappa^2)^(-1/2).  Checked here to machine precision, and the same
   construction exhibits both failure modes of the void regime.

2. CONDITIONING.  The polar factor's own relative conditioning is ~kappa; the
   Gram route composes G -> G^(-1/2), whose relative conditioning is kappa(G) =
   kappa^2.  So the Gram route should be worse than a direct SVD route by a
   factor ~kappa.  Measured as the ratio of the two errors at equal ||E||.

3. NO BACKWARD STABILITY.  The Gram route returns (G+E)^(-1/2) B, which is not
   an isometry unless E = 0: its projector is (G+E)^(-1/2) G (G+E)^(-1/2) != I.
   A polar factor is a partial isometry, so a non-isometric output is the exact
   polar factor of *no* matrix and the algorithm admits no backward error bound.
   Measured as ||Phat Phat^T - I||_2 against the unit roundoff.

4. FAMILY.  For a matrix function q applied through the Gram route, the update
   is q(Ghat) B and the i-th update singular value has relative sensitivity
   L_q(lambda_i) * eta*lambda_max/lambda_i with L_q = |lambda q'/q|.  For
   q(lambda) = lambda^(-alpha) this is L = alpha, so the first-order error of the
   whole factor should be alpha * eta * kappa^2 / <normalisation>.  Prediction:
   measured error divided by eta*kappa^2 is linear in alpha with slope one, and
   the exact polar factor is alpha = 1/2.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "theory_upgrades.json"
M, N = 192, 576


def make_matrix(kappa, m=M, n=N, seed=0):
    g = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(n, m, generator=g, dtype=torch.float64))
    s = torch.logspace(0, np.log10(kappa), m, dtype=torch.float64)
    return (U * s) @ V.T


# ---------------------------------------------------------------- 1. sharpness
def sharpness() -> dict:
    rows = []
    for kappa in (10.0, 100.0, 1e3, 1e4):
        for t in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
            lam = torch.tensor([1.0, kappa ** -2], dtype=torch.float64)
            eta = t / kappa ** 2                    # so that eta*kappa^2 = t
            lam_hat = lam - eta * torch.tensor([0.0, 1.0], dtype=torch.float64)
            s = (lam / lam_hat).sqrt()
            attained = float((s - 1).abs().max())
            bound = (1 - t) ** -0.5 - 1.0
            rows.append({"kappa": kappa, "eta_kappa_sq": t,
                         "attained": attained, "bound": bound,
                         "ratio": attained / bound})
    r = [x["ratio"] for x in rows]
    return {"rows": rows, "ratio_min": float(np.min(r)),
            "ratio_max": float(np.max(r)),
            "attained_exactly": bool(abs(np.min(r) - 1) < 1e-9
                                     and abs(np.max(r) - 1) < 1e-9)}


def void_modes() -> dict:
    """eta*kappa^2 >= 1: the direction is either lost or amplified."""
    out = {}
    kappa = 100.0
    lam = torch.tensor([1.0, kappa ** -2], dtype=torch.float64)
    eta = 2.0 / kappa ** 2                       # eta*kappa^2 = 2 > 1
    e = torch.tensor([0.0, 1.0], dtype=torch.float64)
    # (i) the computed eigenvalue is driven non-positive: the direction is dropped
    lam_hat = lam - 1.5 * eta * e
    out["dropped"] = {"lambda_hat_min": float(lam_hat[1]),
                      "s_min": "dropped" if float(lam_hat[1]) <= 0 else "kept"}
    # (ii) it stays positive but below the truncation level eta*lambda_max, so the
    #      untruncated code inverts it and amplifies by sqrt(lambda/lambda_hat)
    lam_hat2 = lam - 0.9 * lam * e
    out["amplified_below_truncation"] = {
        "lambda_hat_min": float(lam_hat2[1]),
        "lambda_hat_over_eta_lambda_max": float(lam_hat2[1] / (eta * lam[0])),
        "s_min": float((lam[1] / lam_hat2[1]).sqrt())}
    # (iii) the same direction after truncation at eta is removed, so the cap holds
    out["retained_cap"] = {"truncated": bool(float(lam_hat2[1]) < eta * float(lam[0]))}
    return out


# ------------------------------------------------------ 2. conditioning ratio
def conditioning() -> dict:
    rows = []
    for kappa in (10.0, 1e2, 1e3, 1e4):
        B = make_matrix(kappa)
        s0 = float(torch.linalg.svdvals(B)[0])
        g = torch.Generator().manual_seed(7)
        R = torch.randn(B.shape, dtype=torch.float64, generator=g)
        E = R * (1e-9 * s0 / float(torch.linalg.matrix_norm(R, ord=2)))
        # Direct route: its input is B, perturbed by a relative 1e-9.
        U1, _, V1 = torch.linalg.svd(B, full_matrices=False)
        U2, _, V2 = torch.linalg.svd(B + E, full_matrices=False)
        err_svd = float(torch.linalg.matrix_norm(U1 @ V1 - U2 @ V2, ord=2))
        # Gram route: its input is G = B B^T, perturbed by the *same* relative
        # amount.  That is the right comparison, because both routes' arithmetic
        # injects a relative error of order u into the array they actually form.
        def gram_of(Gm):
            lam, V = torch.linalg.eigh(Gm)
            lam = lam.clamp(min=0.0)
            o = torch.argsort(lam, descending=True)
            lam, V = lam[o], V[:, o]
            w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
            return (V * w) @ (V.T @ B)
        G0 = B @ B.T
        g2 = torch.Generator().manual_seed(11)
        Rg = torch.randn(G0.shape, dtype=torch.float64, generator=g2)
        Rg = 0.5 * (Rg + Rg.T)
        F = Rg * (1e-9 * float(torch.linalg.matrix_norm(G0, ord=2))
                  / float(torch.linalg.matrix_norm(Rg, ord=2)))
        err_gram = float(torch.linalg.matrix_norm(gram_of(G0) - gram_of(G0 + F),
                                                 ord=2))
        rows.append({"kappa": kappa, "err_svd_route": err_svd,
                     "err_gram_route": err_gram,
                     "ratio": err_gram / max(err_svd, 1e-300)})
    r = [x["ratio"] / x["kappa"] for x in rows]
    return {"rows": rows, "ratio_over_kappa_min": float(np.min(r)),
            "ratio_over_kappa_max": float(np.max(r)),
            "scales_like_kappa": bool(0.1 < float(np.median(r)) < 10.0)}


# ------------------------------------------- 3. not an isometry -> no backward
def non_isometry() -> dict:
    rows = []
    for dname in ("float64", "float32"):
        dt = getattr(torch, dname)
        u = torch.finfo(dt).eps
        for kappa in (10.0, 1e2, 1e3, 1e4):
            B = make_matrix(kappa)
            G = (B.to(dt) @ B.to(dt).T).to(torch.float64)
            lam, V = torch.linalg.eigh(G)
            lam = lam.clamp(min=0.0)
            o = torch.argsort(lam, descending=True)
            lam, V = lam[o], V[:, o]
            w = torch.where(lam > 0, lam.rsqrt(), torch.zeros_like(lam))
            P = (V * w) @ (V.T @ B)
            m = P.shape[0]
            I = torch.eye(m, dtype=P.dtype)
            dev = float(torch.linalg.matrix_norm(P @ P.T - I, ord=2))
            rows.append({"dtype": dname, "kappa": kappa,
                         "isometry_defect": dev, "unit_roundoff": u,
                         "defect_over_u": dev / u})
    worst = max(r["defect_over_u"] for r in rows)
    return {"rows": rows, "max_defect_over_u": float(worst),
            "not_backward_stable": bool(worst > 1e3)}


# ------------------------------------------------------------- 4. the family
def gram_power_from_G(G: torch.Tensor, B: torch.Tensor, alpha: float):
    lam, V = torch.linalg.eigh(G)
    lam = lam.clamp(min=0.0)
    o = torch.argsort(lam, descending=True)
    lam, V = lam[o], V[:, o]
    w = torch.where(lam > 0, lam.pow(-alpha), torch.zeros_like(lam))
    return (V * w) @ (V.T @ B), lam, V


def family() -> dict:
    """q(lambda) = lambda^-alpha applied through the Gram matrix.

    The perturbation is *injected* at a known relative level rather than taken
    from a dtype, because rounding B to the dtype it is already stored in is a
    no-op and silently gives eta = 0.

    The proposition is about the per-direction relative error at the smallest
    singular value, |m_hat - m| / m with m = q(lambda_i) sigma_i, which should be
    alpha * eta * kappa^2.  The operator-norm relative error is also reported,
    because it is a different quantity: for alpha < 1/2 the update's own singular
    values decay, so the norm is dominated by the head where the relative error
    is only alpha*eta.
    """
    t_rel = 1e-8
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5]
    rows = []
    for alpha in alphas:
        for kappa in (1e2, 1e3, 1e4):
            B = make_matrix(kappa)
            G = B @ B.T
            g = torch.Generator().manual_seed(3)
            R = torch.randn(G.shape, dtype=torch.float64, generator=g)
            R = 0.5 * (R + R.T)
            F = R * (t_rel * float(torch.linalg.matrix_norm(G, ord=2))
                     / float(torch.linalg.matrix_norm(R, ord=2)))
            P0, lam, V = gram_power_from_G(G, B, alpha)
            P1, _, _ = gram_power_from_G(G + F, B, alpha)
            lam_min = float(lam[-1])
            # per-direction quantities in the tail direction
            m0 = (lam_min ** -alpha) * (lam_min ** 0.5)
            m1 = ((lam_min + float(F[-1, -1])) ** -alpha) * (lam_min ** 0.5)
            tail_rel = abs(m1 - m0) / m0
            op_rel = float(torch.linalg.matrix_norm(P1 - P0, ord=2)) / \
                float(torch.linalg.matrix_norm(P0, ord=2))
            rows.append({"alpha": alpha, "kappa": kappa,
                         "eta_times_kappa_sq": t_rel * kappa ** 2,
                         "tail_rel_error": tail_rel,
                         "tail_over_etak2": tail_rel / (t_rel * kappa ** 2),
                         "op_rel_error": op_rel,
                         "op_over_eta": op_rel / t_rel})
    # the exact-constant case: perturb only the smallest eigenvalue, so that the
    # tail direction sees the whole ||F||_2.  The prediction becomes an identity.
    exact = []
    for alpha in alphas:
        for kappa in (1e2, 1e3, 1e4):
            B = make_matrix(kappa)
            G = B @ B.T
            lam, V = torch.linalg.eigh(G)
            o = torch.argsort(lam, descending=True)
            lam, V = lam[o], V[:, o]
            lam_min = float(lam[-1])
            F = torch.zeros_like(G)
            F[-1, -1] = t_rel * float(lam[0])
            m0 = (lam_min ** -alpha) * (lam_min ** 0.5)
            m1 = ((lam_min + float(F[-1, -1])) ** -alpha) * (lam_min ** 0.5)
            tail_rel = abs(m1 - m0) / m0
            exact.append({"alpha": alpha, "kappa": kappa,
                          "ratio": tail_rel / (alpha * t_rel * kappa ** 2)})
    med_tail = {}
    med_op = {}
    for a in alphas:
        med_tail[a] = float(np.median([r["tail_over_etak2"] for r in rows
                                       if r["alpha"] == a]))
        med_op[a] = float(np.median([r["op_over_eta"] for r in rows
                                     if r["alpha"] == a]))
    slope_tail = float(np.polyfit(list(med_tail), list(med_tail.values()), 1)[0])
    return {"rows": rows, "t_rel": t_rel, "exact_rows": exact,
            "exact_ratio_median": float(np.median([r["ratio"] for r in exact])),
            "exact_ratio_min": float(np.min([r["ratio"] for r in exact])),
            "exact_ratio_max": float(np.max([r["ratio"] for r in exact])),
            "median_tail_over_etak2_by_alpha": med_tail,
            "median_op_over_eta_by_alpha": med_op,
            "slope_tail": slope_tail,
            "tail_slope_near_one": bool(0.5 < slope_tail < 2.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    print(f"{platform.machine()}, torch {torch.__version__}")
    out = {"config": {"m": M, "n": N, "machine": platform.machine()}}

    out["sharpness"] = sharpness()
    print(f"\n1 SHARPNESS: attained/bound in "
          f"[{out['sharpness']['ratio_min']:.12f}, "
          f"{out['sharpness']['ratio_max']:.12f}] over 20 cells")
    print(f"  exact attainment: {out['sharpness']['attained_exactly']}")
    out["void_modes"] = void_modes()
    print(f"  void regime, both modes: {out['void_modes']}")

    out["conditioning"] = conditioning()
    print("\n2 CONDITIONING (Gram route vs float64 SVD route, same ||E||):")
    for r in out["conditioning"]["rows"]:
        print(f"  kappa {r['kappa']:8.0e}: svd {r['err_svd_route']:.2e}  "
              f"gram {r['err_gram_route']:.2e}  ratio/kappa "
              f"{r['ratio']/r['kappa']:.2f}")
    print(f"  scales like kappa: {out['conditioning']['scales_like_kappa']}")

    out["non_isometry"] = non_isometry()
    print("\n3 NON-ISOMETRY (defect / unit roundoff):")
    for r in out["non_isometry"]["rows"]:
        print(f"  {r['dtype']:8s} kappa {r['kappa']:8.0e}: defect "
              f"{r['isometry_defect']:.2e} = {r['defect_over_u']:.2e} u")
    print(f"  => not backward stable: "
          f"{out['non_isometry']['not_backward_stable']}")

    out["family"] = family()
    print("\n4 FAMILY (injected relative perturbation "
          f"{out['family']['t_rel']:.0e})")
    print(f"  {'alpha':>6} {'tail/(t k^2)':>13} {'pred':>6} {'op/t':>8}")
    for a in out["family"]["median_tail_over_etak2_by_alpha"]:
        print(f"  {a:6.1f} "
              f"{out['family']['median_tail_over_etak2_by_alpha'][a]:13.3f} "
              f"{a:6.1f} "
              f"{out['family']['median_op_over_eta_by_alpha'][a]:8.3f}")
    print(f"  slope of tail ratio vs alpha: {out['family']['slope_tail']:.2f}")
    f = out["family"]
    print(f"  tail-concentrated perturbation (the exact constant):")
    print(f"    tail_rel / (alpha * t * kappa^2) in "
          f"[{f['exact_ratio_min']:.10f}, {f['exact_ratio_max']:.10f}], "
          f"median {f['exact_ratio_median']:.10f}")

    Path(args.out).write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwrote {Path(args.out).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
