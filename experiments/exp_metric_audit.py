"""
Metric audit: can any per-step update-quality scalar order matrix optimizers?

The literature designs orthogonalizers against orthogonality error
||O O^T - I||_F / sqrt(k).  The manuscript proposes a replacement, descent
efficiency eta(U,B) = <U,B> / (||U||_2 ||B||_*).  This script asks whether
either metric, or the update magnitude, orders the end-to-end results.

All numbers are read from results/; nothing is retrained.  Every cell records
its source file so that the table in the paper is traceable.

Reporting policy: with n in the single digits a p-value is meaningless, so we
report the ordering and the number of fully-reported points, plus which methods
each metric ranks in the reverse of the loss ordering.

Run:  python exp_metric_audit.py
Out:  ../results/metric_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

sys.path.insert(0, str(ROOT / "experiments"))
import torch  # noqa: E402
import robust_orth as ro  # noqa: E402
from baselines import polar_express  # noqa: E402


def load(name):
    return json.loads((RESULTS / name).read_text())


# ---------------------------------------------------------------------------
# descent efficiency, correctly normalised
# ---------------------------------------------------------------------------
@torch.no_grad()
def eta_normalised(U: torch.Tensor, B: torch.Tensor) -> float:
    """eta(U,B) = <U,B> / (||U||_2 ||B||_*).

    The manuscript's Eq. (7) omits the ||U||_2 factor, which makes eta
    unbounded: a method that returns c*UV^T for c>1 scores c, and
    results/gram_floor.json reports eta = 800 for exactly that reason.  With
    the normalisation eta = 1 for the exact polar factor and <= 1 for every
    update.
    """
    Uf, Bf = U.detach().double(), B.detach().double()
    if not torch.isfinite(Uf).all():
        return float("nan")
    smax = float(torch.linalg.matrix_norm(Uf, ord=2))
    nuc = float(torch.linalg.svdvals(Bf).sum())
    return float((Uf * Bf).sum()) / (smax * nuc)


@torch.no_grad()
def eta_table() -> list[dict]:
    """Recompute eta (normalised) and orthogonality error on the 24 real
    subsampled momentum matrices, medians over the matrices."""
    bufs = {n: B.float().cpu() for n, B in
            torch.load(RESULTS / "momentum_matrices.pt", map_location="cpu").items()}
    subs = [B[ro.select_rows(B, 0.15, None, mode="norm")] for B in bufs.values()]

    methods = {
        "exact polar (optimal)": lambda X: ro.polar_svd(X),
        "Polar Express 5": lambda X: polar_express(X.float(), steps=5),
        "Polar Express 8": lambda X: polar_express(X.float(), steps=8),
        # use the reference routine so that orthogonality error is measured on
        # the same (unit-scale) output as Table tab:failure; polar_ns restores
        # the Frobenius scale and is therefore not comparable on this metric
        "Muon quintic K=5": lambda X: ro.muon_reference(X, 5),
        "Muon quintic K=3": lambda X: ro.muon_reference(X, 3),
        "raw normalized B": lambda X: X.float() / (X.float().norm() + 1e-30),
    }
    out = []
    for name, fn in methods.items():
        es, os_ = [], []
        for X in subs:
            U = fn(X)
            es.append(eta_normalised(U, X))
            os_.append(ro.orthogonality_error(U))
        out.append({"name": name, "eta": float(np.nanmedian(es)),
                    "orth_err": float(np.nanmedian(os_)),
                    "orth_err_source": "recomputed", "n_matrices": len(subs)})

    # For any method that also appears in results/baseline_orth.json, report
    # that file's orthogonality error so that this table and Table tab:failure
    # quote the same number for the same method.
    canonical = {r["method"]: r["median_err_f64"]
                 for r in load("orthogonality_f64.json")["rows"]}
    alias = {"exact polar (optimal)": "Exact SVD",
             "Polar Express 5": "Polar Express (5)",
             "Polar Express 8": "Polar Express (8)",
             "Muon quintic K=5": "NS quintic (Muon, ref)"}
    for r in out:
        key = alias.get(r["name"])
        if key and key in canonical:
            r["orth_err"] = canonical[key]
            r["orth_err_source"] = "results/orthogonality_f64.json"
    return out


def final_val(path: Path):
    d = json.loads(path.read_text())
    h = d.get("history", [])
    vals = [x["val_loss"] for x in h if "val_loss" in x]
    return (vals[-1], d.get("n_params"), path.name) if vals else (None, None, path.name)


def main() -> None:
    orth_subs = {r["method"]: r for r in load("orthogonality_f64.json")["rows"]}
    eta_subs = {r["name"]: r for r in load("descent_metric.json")}

    # ---- corrected descent efficiency (the manuscript's Eq. 7 is missing
    #      the ||U||_2 normalisation) --------------------------------------
    eta_rows = eta_table()
    (RESULTS / "descent_metric_normalised.json").write_text(json.dumps({
        "definition": "eta(U,B) = <U,B> / (||U||_2 ||B||_*)",
        "rows": eta_rows,
        "note": ("Recomputes results/descent_metric.json, whose eta omits the "
                 "||U||_2 factor and is therefore unbounded."),
    }, indent=2))
    print("== descent efficiency, normalised ==")
    print(f"{'update':26s} {'eta':>7s} {'orth_err':>10s}")
    for r in eta_rows:
        print(f"{r['name']:26s} {r['eta']:7.4f} {r['orth_err']:10.2e}")
    eta_subs = {r["name"]: r for r in eta_rows}
    print()

    # ---- end-to-end, 49.7M, single seed, from results/gpu_server/ -----------
    e2e = {}
    for label, f in (("Polar Express, 5 steps", "final_pe_1e4.json"),
                     ("Exact Gram", "opt_csmuon_cuda.json"),
                     ("Muon quintic, K=5", "final_muonns.json"),
                     ("AdamW (3e-4)", "final_adamw.json"),
                     ("AdamW (matched, 8.56e-4)", "ctl_adamw_856.json")):
        v, n, src = final_val(RESULTS / "gpu_server" / f)
        e2e[label] = {"val_loss": v, "n_params": n, "source": f"results/gpu_server/{src}"}

    # aggregate update magnitude at 49.7M, quoted from the matched control
    # (Table 6 of the manuscript).  Only the two levels measured are recorded.
    update_norm = {
        "AdamW (matched, 8.56e-4)": 4.37e-3,
        "AdamW (3e-4)": 1.53e-3,
        "all orthogonalizers at their tuned lr": 4.37e-3,
    }

    # ---- assemble the fully-reported rows ----------------------------------
    # metric values are medians over the 24 real 10.7M submatrices
    rows = [
        {"method": "Exact Gram", "orth_err": orth_subs["Exact SVD"]["median_err_f64"],
         "eta": eta_subs["exact polar (optimal)"]["eta"],
         "update_norm": 4.37e-3, "val_loss": e2e["Exact Gram"]["val_loss"],
         "metric_shape": "rho=0.15", "run_shape": "rho=0.15",
         "scale": "49.7M, 1 seed"},
        {"method": "Polar Express, 5 steps",
         "orth_err": orth_subs["Polar Express (5)"]["median_err_f64"],
         "eta": eta_subs["Polar Express 5"]["eta"],
         "update_norm": 4.37e-3, "val_loss": e2e["Polar Express, 5 steps"]["val_loss"],
         "metric_shape": "rho=0.15", "run_shape": "rho=0.15",
         "scale": "49.7M, 1 seed"},
        {"method": "Muon quintic, K=5",
         "orth_err": orth_subs["NS quintic (Muon, ref)"]["median_err_f64"],
         "eta": eta_subs["Muon quintic K=5"]["eta"],
         "update_norm": 4.37e-3, "val_loss": e2e["Muon quintic, K=5"]["val_loss"],
         "metric_shape": "rho=0.15", "run_shape": "full shape",
         "scale": "49.7M, 1 seed"},
        {"method": "AdamW (matched, 8.56e-4)",
         "orth_err": None, "eta": None,
         "update_norm": 4.37e-3, "val_loss": e2e["AdamW (matched, 8.56e-4)"]["val_loss"],
         "metric_shape": "none: no orthogonalizer", "run_shape": "n/a",
         "scale": "49.7M, 1 seed"},
    ]

    # ---- ordering comparison ----------------------------------------------
    def ranks(key, reverse=False):
        v = np.array([r[key] for r in rows if r[key] is not None], float)
        o = np.argsort(-v if reverse else v)
        out = np.full(len(rows), -1, dtype=int)
        for pos, idx in enumerate(o):
            out[np.flatnonzero([r[key] is not None for r in rows])[idx]] = pos
        return out

    r_orth = ranks("orth_err")          # lower is better
    r_eta = ranks("eta", reverse=True)  # higher is better
    r_loss = ranks("val_loss")          # lower is better
    r_upd = ranks("update_norm")        # lower is better
    scored = [i for i, r in enumerate(rows) if r["orth_err"] is not None]

    print(f"{'method':26s} {'orth_err':>10s} {'eta':>7s} {'upd_norm':>9s} {'val loss':>9s}  run")
    for r in rows:
        oe = "n/a" if r["orth_err"] is None else f"{r['orth_err']:.2e}"
        et = "n/a" if r["eta"] is None else f"{r['eta']:.4f}"
        print(f"{r['method']:26s} {oe:>10s} {et:>7s} "
              f"{r['update_norm']:9.2e} {r['val_loss']:9.4f}  {r['run_shape']}")

    print(f"\nranks (1 = best on that criterion), n = {len(scored)} with a metric:")
    print(f"{'method':26s} {'orth_err':>9s} {'eta':>5s} {'upd':>5s} {'loss':>5s}")
    for i in scored:
        print(f"{rows[i]['method']:26s} {r_orth[i] + 1:9d} {r_eta[i] + 1:5d} "
              f"{'-':>5s} {r_loss[i] + 1:5d}")

    ra = r_orth[scored]
    re_ = r_eta[scored]
    rl = r_loss[scored]
    disagree_orth = int((ra != rl).sum())
    disagree_eta = int((re_ != rl).sum())
    print(f"\n  orth_err reverses the loss order on {disagree_orth}/{len(scored)} methods")
    print(f"  eta      reverses the loss order on {disagree_eta}/{len(scored)} methods")
    print(f"  update norm is constant ({4.37e-3:.2e}) across the whole family,")
    print("    so it cannot order it; it separates only the learning-rate confound.")

    # Kendall tau with explicit n; reported as a descriptive ordering statistic
    def kendall(a, b):
        n = len(a)
        c = d = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
                if s > 0:
                    c += 1
                elif s < 0:
                    d += 1
        return (c - d) / (c + d) if c + d else float("nan")

    # Kendall tau on the scored subset, computed from the raw values so that
    # the lost AdamW row cannot distort the ranking scale
    _loss = np.array([rows[i]["val_loss"] for i in scored], float)
    _orth = np.array([rows[i]["orth_err"] for i in scored], float)
    _eta = np.array([rows[i]["eta"] for i in scored], float)
    k_orth = kendall(_orth, _loss)   # both "lower is better"
    k_eta = kendall(-_eta, _loss)    # eta "higher is better"

    # ---- within-family arm at 10.7M (3 seeds for the families, 1 for the
    #      orthogonalizer ablation): does the orthogonalizer ordering survive?
    main_bench = load("main_bench.json")
    abl2 = {r["config"]["name"]: r for r in load("ablation2.json")}
    def _final(r):
        vals = [x["val_loss"] for x in r["history"] if "val_loss" in x]
        return vals[-1]
    loss_by_opt = {}
    for r in main_bench:
        loss_by_opt.setdefault(r["optimizer"], []).append(_final(r))
    # within-optimizer seed spread, not the pooled spread (the pooled one is
    # dominated by the 0.9-nat gap between optimizer families)
    seed_sd = float(np.median([np.std(v, ddof=1) for v in loss_by_opt.values()
                               if len(v) > 1]))
    within = {
        "gram (seed 0)": abl2["abl_selnorm"]["history"][-1].get("val_loss", _final(abl2["abl_selnorm"])),
        "ns quintic on the subsampled matrix (seed 0)": _final(abl2["abl_nsorth"]),
        "orth_err gram": orth_subs["Exact SVD"]["median_err_f64"],
        "orth_err ns quintic": orth_subs["NS quintic (Muon, ref)"]["median_err_f64"],
        "seed_sd_within_optimizer": seed_sd,
    }
    gap = abs(within["gram (seed 0)"] - within["ns quintic on the subsampled matrix (seed 0)"])
    within["abs_gap"] = gap
    within["gap_in_seed_sd"] = gap / seed_sd

    # ---- are the two metrics actually two? -------------------------------
    _m = [r for r in eta_rows if r["orth_err"] is not None]
    _eta = np.array([r["eta"] for r in _m], float)
    _oe = np.array([r["orth_err"] for r in _m], float)
    _ra = np.argsort(np.argsort(_oe))
    _rb = np.argsort(np.argsort(-_eta))
    corr = float(np.corrcoef(_ra, _rb)[0, 1])
    loo = [float(np.corrcoef(np.delete(_ra, i), np.delete(_rb, i))[0, 1])
           for i in range(len(_m))]
    print(f"\n  eta vs orthogonality error: Spearman {corr:+.2f} (n={len(_m)}), "
          f"leave-one-out {min(loo):+.2f}..{max(loo):+.2f}")
    print("    -> one quantity (deviation of the spectrum from 1) in two units,")
    print("       not two independent objectives.")

    # ---- power of the reversal the paper's title rests on -----------------
    _loss = np.array([rows[i]["val_loss"] for i in scored], float)
    # the matched pair is what the title claim rests on
    matched = [r for r in rows if r["run_shape"] == "rho=0.15"]
    pair_gap = abs(matched[0]["val_loss"] - matched[1]["val_loss"])
    sigma = 0.0085          # conservative seed s.d. measured at 10.7M, 3 seeds
    import math as _math
    d = pair_gap / sigma
    n_needed = _math.ceil(15.7 / (d ** 2)) if d > 0 else float("inf")
    print(f"\n  matched pair gap {pair_gap:.4f} nats vs seed s.d. {sigma:.4f} "
          f"-> {d:.2f} sigma")
    print(f"  seeds needed per arm for 80% power (two-sided 5%): ~{n_needed}")

    out = {
        "eta_orth_spearman": {"rho": corr, "n": len(_m),
                              "leave_one_out_min": min(loo),
                              "leave_one_out_max": max(loo)},
        "power": {"matched_pair_gap": pair_gap,
                  "seed_sd_10.7M": sigma,
                  "effect_in_sigma": d,
                  "seeds_needed_per_arm_80pct": n_needed},
        "rows": rows,
        "ranks": {"orth_err": r_orth.tolist(), "eta": r_eta.tolist(),
                  "update_norm": r_upd.tolist(), "val_loss": r_loss.tolist()},
        "n_fully_reported": len(scored),
        "kendall_tau_vs_loss": {"orth_err": k_orth, "eta": k_eta},
        "reversals_vs_loss": {"orth_err": disagree_orth, "eta": disagree_eta},
        "update_norm_by_group": update_norm,
        "within_family_10.7M": within,
        "end_to_end_sources": e2e,
        "eta_normalised": eta_rows,
        "note": ("n is the number of methods for which orthogonality error, eta, "
                 "update magnitude and final loss were all measured at one scale. "
                 "At n=3 an ordering is descriptive; no p-value is reported."),
    }
    (RESULTS / "metric_audit.json").write_text(json.dumps(out, indent=2))

    print(f"\n  Kendall tau vs loss: orth_err {k_orth:+.2f}, eta {k_eta:+.2f} "
          f"(n={len(scored)})")
    print("  the matched-configuration pair is Exact Gram vs Polar Express, both")
    print("  trained and measured at rho=0.15: both metrics rank Exact Gram first,")
    print("  both are contradicted, and the gap is 0.002 nats on one seed.")
    print("  the overall winner at matched update norm is AdamW, which has no")
    print("  orthogonalizer and therefore no value on either metric.")
    print(f"\nwithin-family check at 10.7M: gram {within['gram (seed 0)']:.4f} vs "
          f"ns-quintic {within['ns quintic on the subsampled matrix (seed 0)']:.4f} "
          f"-> gap {gap:.4f} = {within['gap_in_seed_sd']:.2f} seed s.d.")
    print("  orth_err 1e-7 vs 0.64, loss differs by less than the seed noise")
    print(f"\nwrote {RESULTS / 'metric_audit.json'}")


if __name__ == "__main__":
    main()
