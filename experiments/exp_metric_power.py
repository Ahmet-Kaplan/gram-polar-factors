"""
The metric test, with power: 4 orthogonalizers x 3 seeds at 10.7M parameters.

The earlier metric audit had three methods, one seed each, and a 0.002-nat
spread against a 0.007-nat seed standard deviation.  That is not a test; it is a
null result with no power.  This script consumes the 12-run grid in
results/metric_power/ (four orthogonalizers that differ *only* in the
orthogonalizer, identical learning rate and identical unit-RMS update
normalisation, three seeds each) and asks the question properly:

  1. Do the four methods differ at all?  One-way ANOVA, with the between-method
     variance compared against the measured seed variance.  If they do not
     differ, no per-step metric can be validated and we say so.
  2. If they differ, does the metric order the four method means?
  3. What difference could this design have detected?  The minimum detectable
     effect at the observed seed noise, so the null result is bounded rather
     than merely reported.

Run:  python exp_metric_power.py
Out:  ../results/metric_power.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
GRID = RESULTS / "metric_power"
METHODS = ["gram", "ns", "gram_ns", "polar_express"]
SEEDS = [0, 1, 2]


def load(name):
    return json.loads((RESULTS / name).read_text())


def final_loss(path: Path):
    d = json.loads(path.read_text())
    h = d[0]["history"] if isinstance(d, list) else d["history"]
    v = [x["val_loss"] for x in h if "val_loss" in x]
    return (v[-1], d[0] if isinstance(d, list) else d) if v else (None, None)


def main() -> None:
    rows = []
    for m in METHODS:
        losses = []
        for s in SEEDS:
            p = GRID / f"{m}_s{s}.json"
            if not p.exists():
                print(f"  missing {p.name}; run /tmp/metric_power.sh first")
                return
            v, _ = final_loss(p)
            losses.append(float(v))
        rows.append({"method": m, "losses": losses,
                     "mean": float(np.mean(losses)),
                     "sd": float(np.std(losses, ddof=1)),
                     "n": len(losses)})

    # ---- metric values for the same four methods -------------------------
    orth = {r["method"]: r["median_err_f64"]
            for r in load("orthogonality_f64.json")["rows"]}
    eta = {r["name"]: r["eta"]
           for r in load("descent_metric_normalised.json")["rows"]}
    mkey = {"gram": ("Gram eigen (ours)", "exact polar (optimal)"),
            "ns": ("NS quintic (Muon, ref)", "Muon quintic K=5"),
            "gram_ns": ("Gram-NS (Dion3)", None),
            "polar_express": ("Polar Express (5)", "Polar Express 5")}
    for r in rows:
        k = mkey[r["method"]]
        r["orth_err"] = orth.get(k[0])
        r["eta"] = eta.get(k[1]) if k[1] else None

    print(f"{'method':16s} {'orth_err':>10s} {'eta':>7s} {'mean loss':>10s} "
          f"{'sd':>7s}  {'losses'}")
    for r in rows:
        oe = "n/a" if r["orth_err"] is None else f"{r['orth_err']:.2e}"
        et = "n/a" if r["eta"] is None else f"{r['eta']:.4f}"
        print(f"{r['method']:16s} {oe:>10s} {et:>7s} {r['mean']:10.4f} "
              f"{r['sd']:7.4f}  {[round(x, 4) for x in r['losses']]}")

    # ---- one-way ANOVA ---------------------------------------------------
    allv = np.array([x for r in rows for x in r["losses"]], float)
    grand = allv.mean()
    k, n = len(rows), len(SEEDS)
    ss_b = sum(n * (r["mean"] - grand) ** 2 for r in rows)
    ss_w = sum(float(((np.array(r["losses"]) - r["mean"]) ** 2).sum()) for r in rows)
    df_b, df_w = k - 1, k * (n - 1)
    ms_b, ms_w = ss_b / df_b, ss_w / df_w
    F = ms_b / ms_w if ms_w > 0 else float("inf")
    eta2 = ss_b / (ss_b + ss_w) if (ss_b + ss_w) > 0 else float("nan")
    try:
        from scipy.stats import f as fdist, spearmanr
        p_anova = float(fdist.sf(F, df_b, df_w))
    except Exception:
        p_anova = float("nan")
        spearmanr = None

    pooled_sd = float(np.sqrt(ms_w))
    # two-sided 95% CI half-width for a method mean with n=3 (t_{.975,8}=2.306)
    ci = 2.306 * pooled_sd / np.sqrt(n)
    # minimum detectable difference between two method means at 80% power
    mdd = 2.8 * pooled_sd * np.sqrt(2 / n)

    print(f"\nbetween-method variance: MS_between = {ms_b:.3e} (df {df_b})")
    print(f"seed variance:           MS_within  = {ms_w:.3e} (df {df_w}), "
          f"pooled sd = {pooled_sd:.4f}")
    print(f"F({df_b},{df_w}) = {F:.2f}   p = {p_anova:.3g}   eta^2 = {eta2:.3f}")
    print(f"95% CI half-width on a method mean: +/-{ci:.4f}")
    print(f"minimum detectable difference (80% power): {mdd:.4f} nats")
    print(f"observed spread of method means: "
          f"{max(r['mean'] for r in rows) - min(r['mean'] for r in rows):.4f} nats")

    # ---- does the metric order the method means? ------------------------
    res = {"rows": rows,
           "anova": {"k": k, "n_per_method": n, "grand_mean": float(grand),
                     "ms_between": float(ms_b), "ms_within": float(ms_w),
                     "df_between": df_b, "df_within": df_w,
                     "F": float(F), "p": p_anova, "eta_squared": float(eta2),
                     "pooled_sd": pooled_sd, "ci95_halfwidth": float(ci),
                     "min_detectable_difference": float(mdd)}}

    have = [r for r in rows if r["orth_err"] is not None]
    if len(have) >= 3:
        oe = np.array([r["orth_err"] for r in have], float)
        loss = np.array([r["mean"] for r in have], float)
        ra = np.argsort(np.argsort(oe))
        rb = np.argsort(np.argsort(loss))
        rho = float(np.corrcoef(ra, rb)[0, 1])
        res["orth_err_vs_loss"] = {"spearman": rho, "n": len(have)}
        print(f"\north_err vs method-mean loss: Spearman {rho:+.2f} (n={len(have)})")
        if spearmanr is not None:
            res["orth_err_vs_loss"]["scipy_rho"], res["orth_err_vs_loss"]["p"] = \
                [float(x) for x in spearmanr(oe, loss)]
        eta_have = [r for r in have if r["eta"] is not None]
        if len(eta_have) >= 3:
            et = np.array([r["eta"] for r in eta_have], float)
            lo = np.array([r["mean"] for r in eta_have], float)
            re = np.corrcoef(np.argsort(np.argsort(-et)),
                             np.argsort(np.argsort(lo)))[0, 1]
            res["eta_vs_loss"] = {"spearman": float(re), "n": len(eta_have)}
            print(f"eta      vs method-mean loss: Spearman {re:+.2f} (n={len(eta_have)})")

    verdict = ("methods differ beyond seed noise" if p_anova < 0.05
               else "methods do NOT differ beyond seed noise")
    res["verdict"] = verdict
    print(f"\nverdict: {verdict}")
    if p_anova >= 0.05:
        print(f"  the design can only resolve differences above {mdd:.4f} nats "
              f"at 80% power, and the observed spread is "
              f"{max(r['mean'] for r in rows) - min(r['mean'] for r in rows):.4f}")
    (RESULTS / "metric_power.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {RESULTS / 'metric_power.json'}")


if __name__ == "__main__":
    main()
