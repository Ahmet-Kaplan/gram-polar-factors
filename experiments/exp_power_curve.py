"""
How many seeds does an orthogonalizer comparison actually need?

The metric audit (results/metric_power.json) bounds its null result with a
minimum detectable difference (MDD).  The quantity that bound should have been
computed from is the design actually run: a one-way ANOVA with k = 4 arms of
n = 3, not a two-sample test.  The two differ by a factor 1.4 here, and the
difference matters, because the whole claim of that section is "any real effect
is smaller than the MDD".

This script recomputes the bound three ways and turns the variance estimate
around into a sample-size table:

  MDD_global    range of the k means detectable by the one-way ANOVA at 80%
                power (noncentral F, alpha = 0.05);
  MDD_pairwise  difference between any two arms detectable at 80% power
                (noncentral t, df = 2n - 2);
  n_required    arms needed to detect a given difference, including the
                difference actually observed among the four method means.

The within-group s.d. is the pooled estimate from the 12 runs, not a guess, and
the script asserts that the published MDD corresponds to the two-sample normal
approximation -- so the discrepancy is reported, not silently replaced.

Run:  python exp_power_curve.py
Out:  ../results/power_curve.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ALPHA = 0.05
POWER = 0.80


def power_anova(k: int, n: int, f: float) -> float:
    """Power of the one-way ANOVA F test, k groups of n, effect size f (Cohen)."""
    df1, df2 = k - 1, k * (n - 1)
    crit = stats.f.ppf(1 - ALPHA, df1, df2)
    return float(1.0 - stats.ncf.cdf(crit, df1, df2, n * k * f ** 2))


def power_t(n: int, delta: float, sigma: float) -> float:
    """Power of a two-sided two-sample t test, n per arm, df = 2n - 2."""
    df = 2 * n - 2
    crit = stats.t.ppf(1 - ALPHA / 2, df)
    nc = delta / (sigma * np.sqrt(2.0 / n))
    return float(1.0 - stats.nct.cdf(crit, df, nc))


def _bisect(pred, lo=1e-6, hi=1.0, iters=90):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if pred(mid):
            hi = mid
        else:
            lo = mid
    return hi


def mdd_global(k: int, n: int, sigma: float) -> float:
    """Range (max-min) of k equally spaced means detectable at target power."""

    def ok(d):
        f = float(np.std(np.linspace(0.0, d, k)) / sigma)
        return power_anova(k, n, f) >= POWER

    return _bisect(ok)


def mdd_pairwise(n: int, sigma: float) -> float:
    return _bisect(lambda d: power_t(n, d, sigma) >= POWER)


def n_required(k: int, delta: float, sigma: float, nmax: int = 4000) -> int:
    f = float(np.std(np.linspace(0.0, delta, k)) / sigma)
    for n in range(2, nmax + 1):
        if power_anova(k, n, f) >= POWER:
            return n
    return -1


def main() -> None:
    mp = json.loads((RESULTS / "metric_power.json").read_text())
    a = mp["anova"]
    k, n0 = a["k"], a["n_per_method"]
    sigma = a["pooled_sd"]

    means = np.array([r["mean"] for r in mp["rows"]])
    spread = float(means.max() - means.min())
    f_obs = float(np.std(means) / sigma)

    pub = float(a["min_detectable_difference"])
    z = stats.norm.ppf(0.975) + stats.norm.ppf(POWER)
    two_sample_normal = z * sigma * np.sqrt(2.0 / n0)

    g = mdd_global(k, n0, sigma)
    p = mdd_pairwise(n0, sigma)
    n_obs = n_required(k, spread, sigma)

    rows = []
    for delta in (0.005, 0.010, 0.020, 0.030, 0.033, 0.050):
        n = n_required(k, delta, sigma)
        rows.append({"delta_nats": delta,
                     "f": float(np.std(np.linspace(0.0, delta, k)) / sigma),
                     "n_per_arm": n, "total_runs": k * n})

    out = {
        "alpha": ALPHA, "target_power": POWER,
        "k": k, "n_per_arm_reported": n0, "sigma_within": sigma,
        "observed_spread_nats": spread,
        "observed_f": f_obs,
        "published_mdd": pub,
        "published_mdd_recomputed_two_sample_normal": float(two_sample_normal),
        "published_mdd_is_two_sample_normal": bool(abs(pub - two_sample_normal) < 1e-4),
        "published_mdd_power_global": power_anova(
            k, n0, float(np.std(np.linspace(0.0, pub, k)) / sigma)),
        "mdd_global_n3": g,
        "mdd_pairwise_n3": p,
        "n_for_observed_spread": n_obs,
        "runs_for_observed_spread": k * n_obs,
        "table": rows,
        "verdict": (f"the design bounds effects at {g:.3f} nats globally "
                    f"({p:.3f} pairwise), not {pub:.3f}; detecting the observed "
                    f"{spread:.4f}-nat spread needs ~{n_obs} seeds per arm "
                    f"({k * n_obs} runs, {n_obs / n0:.0f}x the runs in use)"),
    }
    (RESULTS / "power_curve.json").write_text(json.dumps(out, indent=2))

    print(f"pooled within-s.d. {sigma:.4f} nats, k={k}, n={n0}")
    print(f"observed spread {spread:.4f} nats, Cohen f={f_obs:.3f}")
    print(f"published MDD {pub:.4f} == two-sample normal {two_sample_normal:.4f} "
          f"({out['published_mdd_is_two_sample_normal']})")
    print(f"published MDD in a 4-arm ANOVA has power "
          f"{out['published_mdd_power_global']:.2f}, not {POWER}")
    print(f"MDD, one-way ANOVA, n=3 : {g:.4f} nats")
    print(f"MDD, pairwise, n=3      : {p:.4f} nats")
    print(f"seeds for the observed spread: {n_obs}/arm ({k * n_obs} runs)")
    print(f"\n{'delta (nats)':>13s} {'f':>7s} {'n/arm':>7s} {'runs':>7s}")
    for r in rows:
        print(f"{r['delta_nats']:13.3f} {r['f']:7.3f} {r['n_per_arm']:7d} "
              f"{r['total_runs']:7d}")
    print(f"\nwrote {RESULTS / 'power_curve.json'}")


if __name__ == "__main__":
    main()
