"""
The descent-efficiency metric is a spectral functional.

The manuscript observes, informally, that on real momentum submatrices the top
ten of 58 directions carry 94% of the nuclear norm, that truncating the exact
polar factor to half the directions retains eta = 0.982, and that a random
subspace gives eta ~ 0.41.  It then declines to claim this as a contribution.

It is one.  Two exact statements hold for every matrix, not just these:

  (T1) Rank-r truncation.  Let B = U diag(s) V^T and B_r its best rank-r
       approximation, with U_r = polar(B_r) = U_r V_r^T.  Then

           eta(U_r, B) = sum_{i<=r} s_i / sum_i s_i        (retained mass)
           orth_err(U_r) = sqrt((m - r) / m)               (support deficit)

       The first depends only on the spectrum's tail, the second only on the
       count r.  They are therefore independent: for every r < m one can choose
       a spectrum with eta = 1 - delta and orth_err = sqrt(1 - r/m), for any
       delta > 0.  No monotone relation between the two metrics can exist.

  (T2) Any spectral method (all polynomials, hence Muon and Polar Express).
       If the update is P(B) = U p(diag(s)) V^T for a scalar map p applied to
       the Frobenius-normalised spectrum, then

           eta(P(B), B) = sum_i s_i p_i / ( max_i |p_i| * sum_i s_i ).

       So eta is computable in O(k) from the spectrum alone -- no matrix
       products, no training run -- and it separates the map p from the
       spectrum it acts on.

This script verifies both against the matrix definitions on the 24 real
momentum matrices, at full shape and at rho = 0.15 row subsampling, and reports
the eta(r) curve.  Agreement is expected at the 1e-12 level; anything looser
means one of the formulas is wrong.

Run:  python exp_eta_spectral.py
Out:  ../results/eta_spectral.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import robust_orth as ro
from baselines import polar_express

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RHO = 0.15
TOL = 1e-10


@torch.no_grad()
def eta_direct(U: torch.Tensor, B: torch.Tensor) -> float:
    """Definition: <U,B> / (||U||_2 ||B||_*)."""
    Uf, Bf = U.detach().double(), B.detach().double()
    smax = torch.linalg.matrix_norm(Uf, ord=2)
    nuc = torch.linalg.svdvals(Bf).sum()
    return float((Uf * Bf).sum() / (smax * nuc))


@torch.no_grad()
def eta_spectral(p: torch.Tensor, s: torch.Tensor) -> float:
    """T2: sum s_i p_i / (max|p| sum s_i)."""
    return float((s * p).sum() / (p.abs().max() * s.sum()))


@torch.no_grad()
def scalar_map(method: str, s: torch.Tensor, K: int) -> torch.Tensor:
    """Apply a method to diag(s) and return the diagonal.

    Every method here is spectral, so acting on a diagonal matrix returns the
    scalar map evaluated at each singular value.  Using the real implementation
    rather than transcribing coefficients keeps the check honest.
    """
    D = torch.diag(s.double())
    if method == "muon_ns":
        return torch.diag(ro.muon_reference(D, K))
    if method == "polar_express":
        return torch.diag(polar_express(D, steps=K))
    raise ValueError(method)


def main() -> None:
    bufs = torch.load(RESULTS / "momentum_matrices.pt", map_location="cpu")
    rows = []
    trunc_rows = []

    for name, B in bufs.items():
        idx = ro.select_rows(B.float(), RHO, None, mode="norm")
        for regime, X in (("full", B.float()), ("sub", B[idx].float())):
            X64 = X.double()
            m, k = X64.shape
            U, s, Vh = torch.linalg.svd(X64, full_matrices=False)
            V = Vh.mT
            mass = s / s.sum()

            # ---- T1: truncation -----------------------------------------
            for r in (1, 2, 5, 10, 20, 29, 58, 116, 192, 384):
                if r > m:
                    continue
                Ur = U[:, :r] @ V[:, :r].mT
                eta_dir = eta_direct(Ur, X64)
                eta_cf = float(mass[:r].sum())
                err = ro.orthogonality_error(Ur)
                err_cf = float(np.sqrt((m - r) / m))
                trunc_rows.append({
                    "param": name, "regime": regime, "m": m, "k": k, "r": r,
                    "eta_direct": eta_dir, "eta_closed_form": eta_cf,
                    "eta_absdiff": abs(eta_dir - eta_cf),
                    "orth_err": err, "orth_err_closed_form": err_cf,
                    "orth_absdiff": abs(err - err_cf),
                })

            # ---- T2: spectral methods -----------------------------------
            for method, K in (("muon_ns", 3), ("muon_ns", 5), ("muon_ns", 10),
                              ("polar_express", 5), ("polar_express", 8)):
                D = torch.diag(s)
                Uout = (ro.muon_reference(D, K) if method == "muon_ns"
                        else polar_express(D, steps=K))
                p = torch.diag(Uout)                 # scalar map on s
                Ufull = U @ Uout @ V.mT              # P(B) = U p(S) V^T
                eta_dir = eta_direct(Ufull, X64)
                eta_sp = eta_spectral(p, s)
                rows.append({
                    "param": name, "regime": regime, "method": method, "K": K,
                    "eta_direct": eta_dir, "eta_spectral": eta_sp,
                    "absdiff": abs(eta_dir - eta_sp),
                    "orth_err": ro.orthogonality_error(Ufull),
                    "p_min": float(p.min()), "p_max": float(p.max()),
                })

    te = np.array([r["eta_absdiff"] for r in trunc_rows])
    to = np.array([r["orth_absdiff"] for r in trunc_rows])
    se = np.array([r["absdiff"] for r in rows])

    print(f"T1 truncation: {len(trunc_rows)} cells, "
          f"max |eta - closed form| = {te.max():.2e}, "
          f"max |err - sqrt(1-r/m)| = {to.max():.2e}")
    print(f"T2 spectral  : {len(rows)} cells, "
          f"max |eta - spectral form| = {se.max():.2e}")

    # eta(r) curve, medians over the 24 submatrices
    curve = []
    for r in (1, 2, 5, 10, 20, 29, 58, 116, 192, 384):
        v = [x for x in trunc_rows if x["regime"] == "sub" and x["r"] == r]
        if not v:
            continue
        curve.append({
            "r": r, "r_over_m": r / 58,
            "eta_median": float(np.median([x["eta_closed_form"] for x in v])),
            "eta_min": float(np.min([x["eta_closed_form"] for x in v])),
            "orth_err_median": float(np.median([x["orth_err"] for x in v])),
        })

    top10 = []
    for name, B in bufs.items():
        idx = ro.select_rows(B.float(), RHO, None, mode="norm")
        s = torch.linalg.svdvals(B[idx].double())
        top10.append(float((s[:10].sum() / s.sum())))
    top10 = np.array(top10)

    # ---- selection rule: norm vs uniform random rows, full matrices -----
    # The manuscript asserts a single number for "a random rank-r subspace"
    # that does not reproduce.  eta depends strongly on r, so the assertion
    # "regardless of r" is false as well.  Measure the comparison the Dion
    # argument actually needs: same number of rows, two selection rules.
    torch.manual_seed(0)
    sel_rows = []
    for name, B in bufs.items():
        X = B.double()
        m, k = X.shape
        order = torch.argsort(X.norm(dim=1), descending=True)
        for r in (10, 29, 58):
            if r > m:
                continue
            for rule in ("norm", "random"):
                perm = order[:r] if rule == "norm" else torch.randperm(m)[:r]
                Uf = torch.zeros_like(X)
                Uf[perm] = ro.polar_svd(X[perm])
                sel_rows.append({"param": name, "r": r, "rule": rule,
                                 "eta": eta_direct(Uf, X)})

    selection = {}
    for r in (10, 29, 58):
        by_rule = {
            rule: {x["param"]: x["eta"] for x in sel_rows
                   if x["r"] == r and x["rule"] == rule}
            for rule in ("norm", "random")
        }
        if not by_rule["norm"]:
            continue
        keys = sorted(by_rule["norm"])
        nrm = np.array([by_rule["norm"][p] for p in keys])
        rnd = np.array([by_rule["random"][p] for p in keys])
        selection[str(r)] = {
            "norm_median": float(np.median(nrm)),
            "random_median": float(np.median(rnd)),
            "median_gap": float(np.median(nrm - rnd)),
            "norm_wins": int((nrm > rnd).sum()),
            "n": len(keys),
        }

    out = {
        "rho": RHO,
        "n_matrices": len(bufs),
        "tolerance": TOL,
        "t1_max_eta_absdiff": float(te.max()),
        "t1_max_orth_absdiff": float(to.max()),
        "t2_max_eta_absdiff": float(se.max()),
        "t1_holds": bool(te.max() < TOL and to.max() < TOL),
        "t2_holds": bool(se.max() < TOL),
        "top10_nuclear_mass_median": float(np.median(top10)),
        "top10_nuclear_mass_min": float(top10.min()),
        "selection": selection,
        "selection_rows": sel_rows,
        "truncation_curve": curve,
        "truncation": trunc_rows,
        "spectral": rows,
    }
    (RESULTS / "eta_spectral.json").write_text(json.dumps(out, indent=2))

    print(f"\nsubmatrices, medians over 24 (m=58):")
    print(f"{'r':>4s} {'r/m':>6s} {'eta':>7s} {'orth_err':>9s}")
    for c in curve:
        print(f"{c['r']:4d} {c['r_over_m']:6.2f} {c['eta_median']:7.4f} "
              f"{c['orth_err_median']:9.3f}")
    print(f"\ntop-10 nuclear mass: median {np.median(top10):.3f}, "
          f"min {top10.min():.3f}")
    print("\nrow selection, full matrices, medians over 24:")
    for r, v in selection.items():
        print(f"  r={r:>3s}  norm {v['norm_median']:.3f}  random "
              f"{v['random_median']:.3f}  gap {v['median_gap']:+.3f}  "
              f"norm wins {v['norm_wins']}/{v['n']}")
    print(f"\nT1 holds: {out['t1_holds']}   T2 holds: {out['t2_holds']}")
    print(f"wrote {RESULTS / 'eta_spectral.json'}")


if __name__ == "__main__":
    main()
