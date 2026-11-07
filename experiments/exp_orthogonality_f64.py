"""
Orthogonality error for every orthogonalizer, measured in float64.

The historical metric (`robust_orth.orthogonality_error`) cast the operand to
float32 before computing ||O O^T - I||_F / sqrt(k), which put a floor of about
1e-7 on the measurement.  Every "exact" entry in the manuscript's tables was
therefore a property of the metric rather than of the method: the exact polar
factor scored 3.0e-7 when its true defect is 4.4e-15, an eight-order inflation.

This script recomputes the table in float64, on the 24 real submatrices at
rho = 0.15, and quantifies the inflation so the old numbers remain auditable.
Timings are carried over from results/baseline_orth.json, which was measured
separately and is unaffected (a timing does not care about the metric).

Run:  python exp_orthogonality_f64.py
Out:  ../results/orthogonality_f64.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import robust_orth as ro
from baselines import gram_newton_schulz, polar_express

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RHO = 0.15


def main() -> None:
    bufs = {n: B.float().cpu() for n, B in
            torch.load(RESULTS / "momentum_matrices.pt", map_location="cpu").items()}
    subs = [B[ro.select_rows(B, RHO, None, mode="norm")] for B in bufs.values()]
    print(f"{len(subs)} submatrices, rho={RHO}, k={subs[0].shape[0]}")

    def clamp_ns(X):
        return ro.polar_ns(X, 5, ro.NS_QUINTIC_PUBLISHED, "frobenius", True)

    methods = {
        "Exact SVD": lambda X: ro.polar_svd(X),
        "Gram eigen (ours)": lambda X: ro.polar_gram(X),
        "Polar Express (5)": lambda X: polar_express(X.float(), steps=5),
        "Polar Express (8)": lambda X: polar_express(X.float(), steps=8),
        "Gram-NS (Dion3)": lambda X: gram_newton_schulz(X.float(), "ise", 5),
        "NS quintic (Muon, ref)": lambda X: ro.muon_reference(X, 5),
        "NS quintic + clamp (ours)": clamp_ns,
    }

    timings = {}
    bp = RESULTS / "baseline_orth.json"
    if bp.exists():
        timings = {r["method"]: r.get("ms") for r in json.loads(bp.read_text())}

    rows = []
    print(f"\n{'method':28s} {'err f64':>10s} {'err f32 metric':>14s} {'inflation':>10s} "
          f"{'smax':>6s} {'smin':>6s}")
    for name, fn in methods.items():
        e64, e32, smax, smin = [], [], [], []
        for X in subs:
            U = fn(X)
            e64.append(ro.orthogonality_error(U))
            e32.append(ro.orthogonality_error(U, dtype=torch.float32))
            s = torch.linalg.svdvals(U.detach().double())
            s = s / s[0]
            smax.append(float(s[0]))
            smin.append(float(s[-1]))
        infl = (np.nanmedian(e32) / np.nanmedian(e64)
                if np.nanmedian(e64) > 0 else float("inf"))
        rows.append({"method": name,
                     "median_err_f64": float(np.nanmedian(e64)),
                     "max_err_f64": float(np.nanmax(e64)),
                     "median_err_f32_metric": float(np.nanmedian(e32)),
                     "metric_inflation": float(infl),
                     "smax_median": float(np.median(smax)),
                     "smin_median": float(np.median(smin)),
                     "ms": timings.get(name)})
        print(f"{name:28s} {np.nanmedian(e64):10.2e} {np.nanmedian(e32):14.2e} "
              f"{infl:10.0e} {np.median(smax):6.3f} {np.median(smin):6.3f}")

    out = {"rho": RHO, "n_submatrices": len(subs),
           "rows": rows,
           "note": ("median_err_f64 is the value to quote. median_err_f32_metric "
                    "reproduces the historical table and shows the measurement "
                    "floor. ms is carried over from results/baseline_orth.json.")}
    (RESULTS / "orthogonality_f64.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'orthogonality_f64.json'}")


if __name__ == "__main__":
    main()
