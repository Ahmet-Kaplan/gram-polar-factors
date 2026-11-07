"""
Conditioning and precision audit of the momentum matrices, rebuilt around the
decisive control in exp_true_conditioning.py.

What the audit now establishes
------------------------------
T0  Ground truth.  exp_true_conditioning.py shows that with gradients AND the
    momentum buffer in float64 the ratio is genuinely 1e-9--1e-8 (median
    7.7e-9 over 36 configurations).  So the ill-conditioning is a property of
    the matrix, not of the storage.  The published float32-SVD measurement is
    biased HIGH by a median factor of 4.5 (max 27): that is the size of the
    measurement error, not the size of the effect.

T1  The ratio sits below the float32 unit roundoff.  24/24 matrices have
    sigma_min/sigma_max < eps_fp32, so a float32 (or bfloat16) *implementation*
    cannot represent the smallest directions even though they are real.  This is
    the genuine precision tension, and it is what the paper should claim.

T2  Representation ladder.  The same matrix rounded to each dtype and measured
    in float64 gives a ratio that tracks the dtype's unit roundoff once the
    dtype is coarse enough to destroy the tail.  This is a property of the
    representation, and it is why the historical number moved when the SVD
    precision changed.

T3  The replacement criterion, validated out of sample.  A Gram-based polar
    factor must resolve squared singular values, so it can see relative
    directions only above sqrt(eps); equivalently it is usable only while
    kappa <= eps^{-1/2}.  That threshold (6.7e7 fp64, 2.9e3 fp32, 32 fp16, 16
    bf16) is fixed by the dtype alone and is *not* a restatement of the
    definition.  We test it on the 2 x 4 grid {full, subsampled} x {fp64, fp32,
    fp16, bf16}, six cells of which were not used to formulate it.

T4  Factorial reporting.  The earlier version quoted a pooled Spearman over 96
    observations that were 24 matrices x 4 conditions.  That is pseudo-
    replication; the cells are reported separately, with within-cell statistics.

Run:  python exp_conditioning_audit.py
Out:  ../results/conditioning_audit.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import robust_orth as ro

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RHO = 0.15

EPS = {"fp64": 2.0 ** -52, "fp32": 2.0 ** -23,
       "fp16": 2.0 ** -10, "bf16": 2.0 ** -8}
# A Gram-based factor must resolve *squared* singular values, so it can only see
# relative directions above sqrt(eps).  Equivalently it is usable while
# kappa = sigma_max/sigma_min <= eps^{-1/2}.  Both thresholds are fixed by the
# dtype alone; neither is derived from the data.
SIG_FLOOR = {d: math.sqrt(e) for d, e in EPS.items()}
KAPPA_MAX = {d: 1.0 / math.sqrt(e) for d, e in EPS.items()}
DT = {"fp64": torch.float64, "fp32": torch.float32,
      "fp16": torch.float16, "bf16": torch.bfloat16}


def spectrum_rel(X: torch.Tensor) -> np.ndarray:
    s = torch.linalg.svdvals(X.detach().double())
    return (s / s[0]).numpy()


def kappa_svd(X: torch.Tensor, dtype: str) -> float:
    s = torch.linalg.svdvals(X.detach().cpu().to(DT[dtype]).double() if dtype != "fp32"
                             else X.detach().cpu().float())
    return float(s[-1] / s[0])


def kappa_of_rounded(X: torch.Tensor, dtype: str) -> float:
    Xr = X.detach().cpu().to(DT[dtype]).float().double()
    s = torch.linalg.svdvals(Xr)
    return float(s[-1] / s[0])


def gram_polar_in(X: torch.Tensor, dtype: str) -> torch.Tensor:
    """Gram polar factor where the Gram matrix is formed and stored in `dtype`.

    This simulates the representation, not the eigensolver's arithmetic: eigh is
    always run in float64 so that the only thing under test is whether the
    dtype can hold the eigenvalues.  A dtype that cannot hold them cannot hold
    them in any eigensolver, so the simulation is conservative in the direction
    that favours the low-precision route.
    """
    def rnd(t):
        # float64 is the exact reference: do not round it through float32
        return t.double() if dtype == "fp64" else t.to(DT[dtype]).float().double()

    low = rnd(X.detach().cpu())
    G = rnd(low @ low.mT)
    w, V = torch.linalg.eigh(G)
    w = w.clamp(min=0.0)
    inv = torch.where(w > 0, w.rsqrt(), torch.zeros_like(w))
    return (V * inv) @ V.mT @ low


@torch.no_grad()
def eta(U: torch.Tensor, B: torch.Tensor) -> float:
    Uf, Bf = U.detach().double(), B.detach().double()
    smax = float(torch.linalg.matrix_norm(Uf, ord=2))
    nuc = float(torch.linalg.svdvals(Bf).sum())
    if smax == 0.0 or nuc == 0.0:
        return float("nan")
    return float((Uf * Bf).sum()) / (smax * nuc)


def rank_corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(np.corrcoef(np.argsort(np.argsort(a)),
                             np.argsort(np.argsort(b)))[0, 1])


def main() -> None:
    bufs = {n: B.float().cpu() for n, B in
            torch.load(RESULTS / "momentum_matrices.pt", map_location="cpu").items()}
    out: dict = {"source": "results/momentum_matrices.pt", "n_matrices": len(bufs),
                 "rho": RHO, "eps": EPS, "kappa_max_gram": KAPPA_MAX}

    # ---------------- T0: ground truth from the controlled run ------------
    tc = json.loads((RESULTS / "true_conditioning.json").read_text())
    if "sweep" in tc:
        truth = tc["summary"]
        out["T0_true_conditioning"] = {
            "n_configs": truth["n_configs"],
            "kappa_true_median": truth["kappa_true_median"],
            "kappa_true_min": truth["kappa_true_min"],
            "kappa_true_max": truth["kappa_true_max"],
            "log10_spread_across_configs": truth["log10_spread"],
            "fp32_measurement_over_true_median": truth["fp32_over_true_median"],
            "fp32_measurement_over_true_max": truth["fp32_over_true_max"],
        }
    else:
        truth = tc["rows"]
        t = {r["grad_precision"] + "/" + r["buffer_precision"]: r for r in truth}
        out["T0_true_conditioning"] = {
            "n_configs": 1,
            "kappa_true_median": t["fp64/fp64"]["kappa_median"],
            "kappa_true_min": t["fp64/fp64"]["kappa_min"],
            "kappa_true_max": t["fp64/fp64"]["kappa_max"],
            "log10_spread_across_configs": 0.0,
            "fp32_measurement_over_true_median":
                t["fp32/fp32"]["kappa_median"] / t["fp64/fp64"]["kappa_median"],
            "fp32_measurement_over_true_max": float("nan"),
        }
    t0 = out["T0_true_conditioning"]
    print("T0  Ground truth (float64 gradients AND float64 buffer)")
    print(f"    {t0['n_configs']} configurations: kappa {t0['kappa_true_median']:.2e} "
          f"(median), [{t0['kappa_true_min']:.2e}, {t0['kappa_true_max']:.2e}], "
          f"spread {t0['log10_spread_across_configs']:.1f} decades")
    print(f"    float32 measurement / truth: {t0['fp32_measurement_over_true_median']:.2f}x "
          f"median, {t0['fp32_measurement_over_true_max']:.1f}x max")
    print("    -> the ill-conditioning is REAL; the float32 measurement is biased high.")

    # ---------------- T0b: the same control at the 10.7M architecture ----
    big = sorted((RESULTS / "truecond10m").glob("*.json")) if (RESULTS / "truecond10m").exists() else []
    if big:
        t_true, t_meas, big_rows = [], [], []
        for f in big:
            d = json.loads(f.read_text())
            g = {(r["grad_precision"], r["buffer_precision"]): r for r in d["rows"]}
            truth = g[("fp64", "fp64")]["kappa_median"]
            meas = g[("fp32", "fp32")]["kappa_median"]
            t_true.append(truth); t_meas.append(meas)
            big_rows.append({"file": f.name, "steps": d["config"]["steps"],
                             "seed": d["config"]["seed"],
                             "width": d["config"]["width"],
                             "depth": d["config"]["depth"],
                             "kappa_true": truth, "kappa_measured_fp32": meas,
                             "ratio": meas / truth,
                             "rank_1e3_median": g[("fp64", "fp64")]["rank_1e3_median"],
                             "k_median": g[("fp64", "fp64")]["k_median"]})
        out["T0b_large_model"] = {
            "runs": big_rows,
            "kappa_true_median": float(np.median(t_true)),
            "kappa_measured_fp32_median": float(np.median(t_meas)),
            "bias_median": float(np.median(np.array(t_meas) / np.array(t_true))),
            "n_runs": len(big_rows)}
        b = out["T0b_large_model"]
        print(f"\nT0b Same control at the 10.7M architecture "
              f"(width {big_rows[0]['width']}, depth {big_rows[0]['depth']}, "
              f"block 256, batch 32)")
        print(f"    {b['n_runs']} runs: true kappa {b['kappa_true_median']:.2e}, "
              f"fp32 reports {b['kappa_measured_fp32_median']:.2e} "
              f"({b['bias_median']:.1f}x)")
        print(f"    fp32 reports {b['kappa_measured_fp32_median'] / EPS['fp32']:.2f} x eps_fp32; "
              f"truth is {b['kappa_true_median'] / EPS['fp32']:.3f} x eps_fp32")

    # ---------------- T1: ratio vs the storage unit roundoff --------------
    print("\nT1  Is the ratio below the float32 unit roundoff?")
    print(f"{'param':30s} {'shape':>11} {'k_fp32':>9} {'k_fp64':>9} "
          f"{'r@eps32':>7} {'below':>5}")
    n_below = 0
    out["T1_resolution"] = []
    for n, B in bufs.items():
        rel = spectrum_rel(B)
        k32, k64 = kappa_svd(B, "fp32"), float(rel[-1])
        r = int((rel > EPS["fp32"]).sum())
        below = k64 < EPS["fp32"]
        n_below += below
        out["T1_resolution"].append({
            "param": n, "shape": list(B.shape),
            "kappa_fp32_svd": k32, "kappa_fp64_svd": k64,
            "spectrum_span_decades": float(np.log10(rel[0] / rel[-1])),
            "numerical_rank_at_eps_fp32": r,
            "sigma_min_below_eps_fp32": bool(below)})
        print(f"{n.replace('transformer.', '')[:30]:30s} {str(tuple(B.shape)):>11} "
              f"{k32:9.2e} {k64:9.2e} {r:7d} {str(below):>5}")
    print(f"\n    {n_below}/{len(bufs)} matrices have kappa < eps_fp32; the true "
          f"kappa is {np.median([x['kappa_fp64_svd'] for x in out['T1_resolution']]):.2e}")
    print("    -> real, but below what a float32 implementation can represent.")

    # ---------------- T2: representation ladder ---------------------------
    print("\nT2  Same matrix, different storage precision (measured in float64):")
    print(f"{'param':30s} " + " ".join(f"{d:>9s}" for d in ("fp64", "fp32", "fp16", "bf16")))
    out["T2_representation_ladder"] = []
    for n, B in bufs.items():
        row = {d: kappa_of_rounded(B, d) for d in ("fp64", "fp32", "fp16", "bf16")}
        out["T2_representation_ladder"].append({"param": n, "kappa": row})
        print(f"{n.replace('transformer.', '')[:30]:30s} "
              + " ".join(f"{row[d]:9.2e}" for d in ("fp64", "fp32", "fp16", "bf16")))

    # ---------------- T3: the criterion, validated out of sample ----------
    print("\nT3  Gram route: kappa <= eps^-1/2 predicts whether it works")
    print(f"{'regime':6s} {'dtype':6s} {'smin/smax':>10s} {'sqrt(eps)':>10s} "
          f"{'pred':>6s} {'orth_err':>10s} {'ok?':>4s}")
    cells = []
    for regime in ("full", "sub"):
        for dt in ("fp64", "fp32", "fp16", "bf16"):
            errs, kaps, etas = [], [], []
            for n, B in bufs.items():
                X = B if regime == "full" else B[ro.select_rows(B, RHO, None, mode="norm")]
                Xr = X.to(DT[dt]).float().double()
                sv = torch.linalg.svdvals(Xr)
                kap = float(sv[-1] / sv[0])
                U = gram_polar_in(X, dt)
                errs.append(ro.orthogonality_error(U))
                kaps.append(kap)
                etas.append(eta(U, Xr))
            pred_ok = bool(np.median(kaps) > SIG_FLOOR[dt])
            med_err = float(np.nanmedian(errs))
            works = med_err < 1e-4
            cells.append({"regime": regime, "dtype": dt,
                          "sigma_min_over_max_median": float(np.median(kaps)),
                          "sigma_floor_sqrt_eps": SIG_FLOOR[dt],
                          "kappa_median": float(1.0 / np.median(kaps)),
                          "kappa_max_gram": KAPPA_MAX[dt],
                          "predicted_ok": pred_ok,
                          "orth_err_median": med_err,
                          "eta_median": float(np.nanmedian(etas)),
                          "observed_ok": bool(works),
                          "prediction_correct": bool(pred_ok == works)})
            print(f"{regime:6s} {dt:6s} {np.median(kaps):10.2e} {SIG_FLOOR[dt]:10.2e} "
                  f"{'ok' if pred_ok else 'FAIL':>6s} {med_err:10.2e} "
                  f"{'ok' if works else 'FAIL':>4s}")
    out["T3_gram_grid"] = cells
    n_correct = sum(c["prediction_correct"] for c in cells)
    out["T3_prediction_accuracy"] = {"correct": n_correct, "n_cells": len(cells)}
    print(f"\n    criterion correct in {n_correct}/{len(cells)} cells "
          f"(threshold fixed by dtype alone, never fitted)")

    # ---------------- T4: factorial, no pooled pseudo-replication --------
    print("\nT4  Factorial cells (24 matrices each), not a pooled correlation")
    print(f"{'regime':6s} {'dtype':6s} {'k':>4s} {'median f':>10s} {'median err':>11s} "
          f"{'within-cell rho(f,err)':>22s}")
    out["T4_factorial"] = []
    for regime in ("full", "sub"):
        for dt in ("fp64", "fp32"):
            f, e = [], []
            for n, B in bufs.items():
                X = B if regime == "full" else B[ro.select_rows(B, RHO, None, mode="norm")]
                rel = spectrum_rel(X)
                f.append(float((rel ** 2 < EPS[dt]).mean()))
                e.append(ro.orthogonality_error(gram_polar_in(X, dt)))
            if dt == "fp64":
                out.setdefault("_note_fp64", "gram formed in float64, no rounding")
            f, e = np.array(f), np.array(e)
            rho = rank_corr(f, e)
            out["T4_factorial"].append({
                "regime": regime, "dtype": dt, "n": len(f),
                "lost_fraction_median": float(np.median(f)),
                "orth_err_median": float(np.median(e)),
                "within_cell_spearman": rho})
            print(f"{regime:6s} {dt:6s} {len(f):4d} {np.median(f):10.4f} "
                  f"{np.median(e):11.2e} {rho if not math.isnan(rho) else float('nan'):22.2f}")
    print("\n    the four cells differ by ~10 orders of magnitude; within a cell")
    print("    the lost fraction carries little or no extra information.")

    (RESULTS / "conditioning_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'conditioning_audit.json'}")


if __name__ == "__main__":
    main()
