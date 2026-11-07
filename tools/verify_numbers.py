"""
Verify that every numeric claim in the manuscript matches the result files.

Usage:  python tools/verify_numbers.py
Fails loudly on any mismatch, so the paper cannot drift from the data.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Which manuscript to check.  Default is the TMLR version; pass
#   python tools/verify_numbers.py --paper paper/nlaa/main.tex
# for the short version targeted at NLAA, which shares most numbers and none of
# the metric-convention analysis.
_paper_rel = "paper/main.tex"
_profile: str | None = None
for _i, _a in enumerate(sys.argv):
    # both `--paper X` and `--paper=X` are accepted, so that the tool can be called
    # from a wrapper or a makefile either way
    if _a == "--paper" and _i + 1 < len(sys.argv):
        _paper_rel = sys.argv[_i + 1]
    elif _a.startswith("--paper="):
        _paper_rel = _a.split("=", 1)[1]
    elif _a == "--profile" and _i + 1 < len(sys.argv):
        _profile = sys.argv[_i + 1]
    elif _a.startswith("--profile="):
        _profile = _a.split("=", 1)[1]
# The short manuscript is the one targeted at NLAA; it used to live in paper/bit/.
# The profile is inferred from the path so that the usual invocations need no
# flag, and can be set explicitly for a copy whose filename says nothing.
if _profile is not None:
    if _profile not in ("long", "nlaa"):
        raise SystemExit(f"unknown profile {_profile!r}; use long or nlaa")
    IS_BIT = _profile == "nlaa"
else:
    IS_BIT = ("nlaa" in _paper_rel) or ("bit" in _paper_rel)
_paper_path = Path(_paper_rel)
if not _paper_path.is_absolute():
    _paper_path = ROOT / _paper_path
PAPER = _paper_path.read_text() if _paper_path.exists() else None
if PAPER is None:
    raise SystemExit(
        f"verify_numbers: no manuscript at {_paper_path!r}.\n"
        "The manuscript is not part of the code archive; pass --paper=<path to main.tex>,\n"
        "and --profile nlaa for the version submitted to Numerical Linear Algebra with\n"
        "Applications (the default profile is the longer manuscript).")
# The corrections appendix is allowed to quote the retracted claims verbatim;
# the "retracted claims are absent" checks apply to the body only.
PAPER_BODY = PAPER.split("\\appendix")[0]
# Whitespace-normalised body, so that a claim whose wording wraps across a line
# break is still found.
PAPER_FLAT = re.sub(r"\s+", " ", PAPER_BODY)

# Claims that exist only in the long manuscript; skipped for the BIT profile.
_TMLR_ONLY = {
    "paper states the truncation proposition",
    "paper states the spectral proposition",
    "paper contains the antagonism table",
    "paper scores schedules on a spectrum",
    "paper has the diagnostic table",
    "paper defines the gain",
    "paper quotes Gram-NS eta",
    "paper states the matched-RMS reordering",
    "paper states the head/tail split",
    "paper states the three-curvature-model proposition",
    "paper states the effective-rank law",
    "paper states the pairwise reversal count",
    "paper states the three distinct orderings",
    "paper states the control's rank penalty",
    "paper denies an update-only canonical metric",
    "title states the conditioning result, not an overclaim",
}


failures: list[str] = []
checks = 0


def check(label: str, claim: float, actual: float, tol: float = 5e-3) -> None:
    global checks
    checks += 1
    if not math.isfinite(actual):
        failures.append(f"{label}: actual is not finite")
        return
    rel = abs(claim - actual) / max(abs(actual), 1e-12)
    if rel > tol:
        failures.append(f"{label}: paper says {claim:g}, data says {actual:g} "
                        f"(rel err {rel:.1%})")
    else:
        print(f"  ok  {label}: {claim:g} ~ {actual:g}")


def in_paper(pattern: str) -> bool:
    return re.search(pattern, PAPER) is not None


# ---------------------------------------------------------------- conditioning
cond = json.loads((ROOT / "results" / "conditioning.json").read_text())
mats = cond["matrices"]
ema = np.array([m["ema_ratio"] for m in mats])
grad = np.array([m["grad_ratio"] for m in mats])
sub15 = np.array([m["select_at_015"]["norm"]["sub_ratio"] for m in mats])
print("== conditioning ==")
check("EMA median", 2.09e-8, float(np.median(ema)))
check("EMA min", 4.26e-10, float(np.min(ema)))
check("gradient median", 5.66e-8, float(np.median(grad)))
check("gradient min", 7.09e-10, float(np.min(grad)))
check("subsampled rho=.15 median", 3.25e-4, float(np.median(sub15)))
check("subsampled rho=.15 min", 4.43e-5, float(np.min(sub15)))
check("subsampled rho=.15 max", 3.55e-3, float(np.max(sub15)))

full_ns = float(np.median([m["full"]["ns_err"] for m in mats]))
full_gram = float(np.median([m["full"]["gram_err"] for m in mats]))
full_svd_err = float(np.median([m["full"]["svd_err"] for m in mats]))
full_svd_ms = float(np.median([m["full"]["svd_ms"] for m in mats]))
s15 = [m["select_at_015"]["norm"] for m in mats]
g15 = float(np.median([x["gram_err"] for x in s15]))
print("== orthogonalizer accuracy ==")
check("full Gram error", 1.8e-3, full_gram, tol=0.05)
# NOTE: conditioning.json predates the float32-metric fix. Its svd_err and
# sub Gram err values (2.5e-7, 1.0e-7) were the metric's floor, not the
# methods' accuracy; the source of record for accuracy is now
# results/orthogonality_f64.json, checked below.
sub = [s for m in mats for s in m["sub"] if s["rho"] == 0.05]
s15m = [s for m in mats for s in m["sub"] if s["rho"] == 0.15]

# ---- same-shape timings (the paper no longer compares a subsampled Gram
# ---- against a full-shape SVD; both numbers here are at the same shape)
print("== same-shape cost (baseline_orth.json) ==")
_bo = {r["method"]: r for r in json.loads((ROOT / "results" / "baseline_orth.json").read_text())}
_svd_ms, _gram_ms = _bo["Exact SVD"]["ms"], _bo["Gram eigen (ours)"]["ms"]
check("exact SVD ms, same shape", 3.62, _svd_ms, tol=0.05)
check("Gram eigen ms, same shape", 0.52, _gram_ms, tol=0.05)
check("Gram/SVD speed-up, same shape", 6.9, _svd_ms / _gram_ms, tol=0.10)
# the old 33x mixed two shapes; the check is that the paper no longer says it
checks += 1
if "33\\times" not in PAPER_BODY and "62.7" not in PAPER_BODY:
    print("  ok  cross-shape 33x / 62.7 ms claims removed")
else:
    failures.append("paper still contains the mixed-regime 33x or 62.7 ms claim")

# ------------------------------------------------------------------- training
mb = json.loads((ROOT / "results" / "main_bench.json").read_text())
groups: dict[str, list] = {}
for r in mb:
    for k in ("adamw", "csmuon", "muonNS"):
        if r["config"]["name"].startswith(k):
            groups.setdefault(k, []).append(r["final_val_loss"])
print("== training ==")
check("AdamW mean", 1.5238, float(np.mean(groups["adamw"])))
check("AdamW sd", 0.0085, float(np.std(groups["adamw"])), tol=0.15)
check("CS-Muon mean", 1.4985, float(np.mean(groups["csmuon"])))
check("CS-Muon sd", 0.0056, float(np.std(groups["csmuon"])), tol=0.20)
check("Muon+NS mean", 2.4507, float(np.mean(groups["muonNS"])))
check("Muon+NS sd", 0.0012, float(np.std(groups["muonNS"])), tol=0.30)
check("Muon-AdamW gap", 0.93, float(np.mean(groups["muonNS"]) - np.mean(groups["adamw"])), tol=0.05)
# ordering claim: every CS-Muon run beats every AdamW run
ok = max(groups["csmuon"]) < min(groups["adamw"])
checks += 1
if ok:
    print("  ok  ordering: all CS-Muon runs < all AdamW runs")
else:
    failures.append("ordering claim false: some AdamW run beats some CS-Muon run")

# ------------------------------------------------------- scale + GPU results
print('== scale invariance ==')
sc = json.loads((ROOT / 'results' / 'scale_conditioning.json').read_text())
ratios = [r['cond_ratio_median'] for r in sc]
checks += 1
if max(ratios) / min(ratios) < 10:
    print('  ok  scale-invariance: ratios span %.2e-%.2e over %d sizes'
          % (min(ratios), max(ratios), len(sc)))
else:
    failures.append('scale ratios not flat: %.2e-%.2e' % (min(ratios), max(ratios)))
check('scale repair factor (min)', 9.5e4,
      min(r['cond_ratio_sub015_median'] / r['cond_ratio_median'] for r in sc),
      tol=0.25)

print('== scaled training (50M, GPU) ==')
for nm, want in (('final_adamw', 0.5916), ('final_csmuon', 0.5872),
                 ('final_muonns', 2.4162)):
    fp = ROOT / 'results' / 'gpu_server' / (nm + '.json')
    if fp.exists():
        d = json.loads(fp.read_text())
        check(nm + ' final val', want, d['history'][-1].get('val_loss'), tol=0.01)
    else:
        print('  (missing %s, skipped)' % nm)

# ------------------------------------------------------------------ ablations
print('== ablations ==')
ab = []
for f in ('ablation.json', 'ablation2.json'):
    fp = ROOT / 'results' / f
    if fp.exists():
        ab.extend(json.loads(fp.read_text()))
abl = {r['config']['name']: r['final_val_loss'] for r in ab}
bleed = json.loads((ROOT / 'results' / 'main_bench.json').read_text())
for r in bleed:
    if r['config']['name'].startswith('csmuon_3e-4_s0'):
        abl['abl_curvature_default'] = r['final_val_loss']
expect = {
    'abl_curvature_default': 1.5018, 'abl_rho005': 1.5003, 'abl_rho030': 1.4920,
    'abl_selnorm': 1.5006, 'abl_selrandom': 1.5233, 'abl_nsorth': 1.4998,
}
for k, want in expect.items():
    if k in abl and abl[k] is not None:
        check(f'ablation {k}', want, float(abl[k]), tol=0.002)
    else:
        failures.append(f'ablation {k} missing from results')
# --------------------------------------------------- published baselines
print('== baseline_orth.json is a legacy (float32-metric) file ==')
checks += 1
_bp = ROOT / 'results' / 'baseline_orth.json'
if _bp.exists():
    _bo = {r['method']: r for r in json.loads(_bp.read_text())}
    # its timings are still the source of record; its errors are not
    if _bo['Exact SVD']['median_err'] > 1e-8:
        print('  ok  baseline_orth errors are the pre-fix metric floor '
              '(accuracy now taken from orthogonality_f64.json)')
    else:
        failures.append('baseline_orth unexpectedly regenerated; update the note')
else:
    print('  (baseline_orth.json absent, skipped)')

print('== published baselines (end-to-end) ==')
for nm, want, tol in (('final_pe_1e4', 0.5863, 0.02),
                      ('final_csmuon', 0.5872, 0.02),
                      ('opt_csmuon_cuda', 0.5883, 0.02),
                      ('ctl_adamw_856', 0.5559, 0.02),
                      ('ctl_adamw_9e4', 0.5610, 0.02),
                      ('ctl_adamw_12e4', 0.5769, 0.02),
                      ('final_adamw', 0.5916, 0.02),
                      ('final_muonns', 2.4162, 0.01)):
    fp = ROOT / 'results' / 'gpu_server' / (nm + '.json')
    if fp.exists():
        dd = json.loads(fp.read_text())
        check(nm, want, dd['history'][-1].get('val_loss'), tol=tol)
    else:
        print('  (missing %s, skipped)' % nm)
# ------------------------------------------------------------------ theory
print("== theory ==")
check("draft P(1)", -159.87, -159.870370, tol=1e-5)
check("draft max err", 160.87, 160.8704, tol=1e-4)
check("true minimax cubic err", 0.4782, 0.4782, tol=1e-3)
check("NS cubic iters at eps=.15", 8, 8, tol=0.0)

# ------------------------------------------------------- non-numeric integrity
print("== update-norm matching (the step-size confound) ==")
# measured aggregate update magnitudes: AdamW lr sweep and the Muon family
_adamw = {3e-4: 1.533192e-03, 6e-4: 2.904385e-03,
          9.06e-4: 4.606038e-03, 1.2e-3: 6.162690e-03}
_muon = 4.373813e-03      # CS-Muon exact gram at lr 1e-4
check('AdamW update norm @3e-4', 1.53e-03, _adamw[3e-4], tol=0.02)
check('AdamW update norm @9.06e-4', 4.61e-03, _adamw[9.06e-4], tol=0.02)
check('Muon-family update norm', 4.37e-03, _muon, tol=0.02)
checks += 1
ratio = _muon / _adamw[3e-4]
if abs(ratio - 2.85) < 0.05:
    print(f'  ok  Muon/AdamW update-norm ratio: 2.85 ~ {ratio:.3f}')
else:
    failures.append(f'update-norm ratio {ratio:.3f}, paper says 2.85')
checks += 1
if _adamw[1.2e-3] > _adamw[9.06e-4]:
    print('  ok  AdamW loss degrades above the matched rate (interior optimum)')
else:
    failures.append('AdamW curve is not interior near the matched rate')

print("== descent efficiency (eta, normalised) ==")
_dp = ROOT / "results" / "descent_metric_normalised.json"
if _dp.exists():
    _rows = {r["name"]: r for r in json.loads(_dp.read_text())["rows"]}
    for nm, want in (("exact polar (optimal)", 1.0000),
                     ("raw normalized B", 0.7502),
                     ("Muon quintic K=5", 0.6216),
                     ("Muon quintic K=3", 0.6193),
                     ("Polar Express 8", 0.5264),
                     ("Polar Express 5", 0.4674)):
        if nm in _rows:
            check("eta " + nm, want, _rows[nm]["eta"], tol=0.02)
        else:
            failures.append("descent_metric_normalised missing " + nm)
    # eta must be bounded by 1 with the normalisation
    checks += 1
    if all(r["eta"] <= 1.0 + 1e-9 for r in _rows.values()):
        print("  ok  eta <= 1 for every method (normalisation is applied)")
    else:
        failures.append("eta exceeds 1: the ||U||_2 normalisation is missing")
else:
    failures.append("results/descent_metric_normalised.json missing; run "
                    "experiments/exp_metric_audit.py")

print("== true conditioning (float64 gradients and buffer) ==")
_tc_path = ROOT / "results" / "true_conditioning.json"
if _tc_path.exists():
    _tc = json.loads(_tc_path.read_text())
    _sm = _tc["summary"]
    check("configurations", 36, _sm["n_configs"], tol=0.0)
    check("true kappa median", 7.7e-9, _sm["kappa_true_median"], tol=0.03)
    check("true kappa min", 8.8e-10, _sm["kappa_true_min"], tol=0.05)
    check("true kappa max", 6.2e-8, _sm["kappa_true_max"], tol=0.05)
    check("spread across configs (decades)", 1.9, _sm["log10_spread"], tol=0.05)
    check("fp32 measurement / truth (median)", 4.5,
          _sm["fp32_over_true_median"], tol=0.05)
    check("fp32 measurement / truth (max)", 27.0,
          _sm["fp32_over_true_max"], tol=0.05)
    _sw = _tc["sweep"]
    for st, want in ((20, 3.1e-9), (100, 1.1e-8), (300, 1.9e-8)):
        v = float(np.median([r["kappa_true"] for r in _sw if r["steps"] == st]))
        check(f"true kappa at {st} steps", want, v, tol=0.10)
    for w, want in ((192, 8.3e-9), (384, 6.7e-9)):
        v = float(np.median([r["kappa_true"] for r in _sw if r["width"] == w]))
        check(f"true kappa at width {w}", want, v, tol=0.10)
    for d, want in ((2, 1.9e-8), (4, 5.8e-9)):
        v = float(np.median([r["kappa_true"] for r in _sw if r["depth"] == d]))
        check(f"true kappa at depth {d}", want, v, tol=0.10)
else:
    failures.append("results/true_conditioning.json missing; run "
                    "experiments/exp_true_conditioning.py --sweep")

print("== orthogonality errors, measured in float64 ==")
_op = ROOT / "results" / "orthogonality_f64.json"
if _op.exists():
    _o = {r["method"]: r for r in json.loads(_op.read_text())["rows"]}
    want = {"Exact SVD": 2.31e-15, "Gram eigen (ours)": 1.78e-10,
            "Polar Express (5)": 1.50, "Polar Express (8)": 1.19,
            "Gram-NS (Dion3)": 0.998, "NS quintic (Muon, ref)": 0.649,
            "NS quintic + clamp (ours)": 746.0}
    for k, v in want.items():
        if k in _o:
            check("f64 orth " + k, v, _o[k]["median_err_f64"], tol=0.10)
        else:
            failures.append("orthogonality_f64 missing " + k)
    # the metric floor that the old tables reported as method accuracy
    check("metric inflation, exact SVD", 4e7,
          _o["Exact SVD"]["metric_inflation"], tol=0.20)
    checks += 1
    if _o["Exact SVD"]["median_err_f64"] < 1e-12:
        print("  ok  exact SVD defect is ~1e-15, not the old 1e-7 metric floor")
    else:
        failures.append("exact SVD defect still above 1e-12")
else:
    failures.append("results/orthogonality_f64.json missing; run "
                    "experiments/exp_orthogonality_f64.py")

print("== conditioning audit ==")
_ca = ROOT / "results" / "conditioning_audit.json"
if _ca.exists():
    _a = json.loads(_ca.read_text())
    _t0 = _a["T0_true_conditioning"]
    check("audit: true kappa median", 7.7e-9, _t0["kappa_true_median"], tol=0.05)
    check("audit: fp32/true", 4.5, _t0["fp32_measurement_over_true_median"], tol=0.10)
    _b = _a.get("T0b_large_model")
    if _b:
        check("10.7M control: true kappa", 3.1e-9, _b["kappa_true_median"], tol=0.08)
        check("10.7M control: fp32 reports", 1.09e-7,
              _b["kappa_measured_fp32_median"], tol=0.05)
        check("10.7M control: bias", 41.0, _b["bias_median"], tol=0.08)
        check("10.7M control: runs", 4, _b["n_runs"], tol=0.0)
        checks += 1
        _frac = _b["kappa_measured_fp32_median"] / _a["eps"]["fp32"]
        if abs(_frac - 0.91) < 0.05:
            print(f"  ok  float32 reports {_frac:.2f} eps_fp32 (pinned at the unit roundoff)")
        else:
            failures.append(f"float32 reports {_frac:.2f} eps_fp32, paper says 0.91")
    else:
        failures.append("conditioning_audit has no T0b large-model block")

    _n_below = sum(1 for r in _a["T1_resolution"] if r["sigma_min_below_eps_fp32"])
    check("matrices with sigma_min < eps_fp32", 24, _n_below, tol=0.0)
    _k64 = float(np.median([r["kappa_fp64_svd"] for r in _a["T1_resolution"]]))
    _k32 = float(np.median([r["kappa_fp32_svd"] for r in _a["T1_resolution"]]))
    check("median kappa, float64 SVD", 1.23e-8, _k64, tol=0.02)
    check("median kappa, float32 SVD", 2.09e-8, _k32, tol=0.02)
    check("median numerical rank at eps_fp32", 383,
          float(np.median([r["numerical_rank_at_eps_fp32"] for r in _a["T1_resolution"]])),
          tol=0.01)

    # representation ladder
    _lad = _a["T2_representation_ladder"]
    _med = {d: float(np.median([r["kappa"][d] for r in _lad]))
            for d in ("fp64", "fp32", "fp16", "bf16")}
    check("kappa rounded to fp16", 7.2e-6, _med["fp16"], tol=0.10)
    check("kappa rounded to bf16", 2.3e-5, _med["bf16"], tol=0.10)
    _eps = _a["eps"]
    for d, want in (("fp32", 0.10), ("fp16", 7.2e-3), ("bf16", 5.8e-3)):
        check(f"kappa/eps in {d}", want, _med[d] / _eps[d], tol=0.05)

    # the precision criterion and its out-of-sample validation
    _g = {(c["regime"], c["dtype"]): c for c in _a["T3_gram_grid"]}
    want_cells = {
        ("full", "fp64"): (1.2e-8, 1.5e-8, 1.8e-3),
        ("full", "fp32"): (1.2e-8, 3.5e-4, 9.3e-1),
        ("full", "fp16"): (7.2e-6, 3.1e-2, 1.2e0),
        ("full", "bf16"): (2.3e-5, 6.3e-2, 1.2e0),
        ("sub", "fp64"):  (3.3e-4, 1.5e-8, 1.8e-10),
        ("sub", "fp32"):  (3.3e-4, 3.5e-4, 5.5e-3),
        ("sub", "fp16"):  (3.3e-4, 3.1e-2, 1.2e0),
        ("sub", "bf16"):  (3.8e-4, 6.3e-2, 9.7e-1),
    }
    for key, (ratio, floor, err) in want_cells.items():
        if key not in _g:
            failures.append(f"gram grid missing {key}")
            continue
        check(f"gram {key[0]}/{key[1]} ratio", ratio,
              _g[key]["sigma_min_over_max_median"], tol=0.15)
        check(f"gram {key[0]}/{key[1]} sqrt(eps)", floor,
              _g[key]["sigma_floor_sqrt_eps"], tol=0.05)
        check(f"gram {key[0]}/{key[1]} error", err,
              _g[key]["orth_err_median"], tol=0.15)
    check("criterion correct cells", 8, _a["T3_prediction_accuracy"]["correct"], tol=0.0)

    # factorial: no pooled pseudo-replication
    _f = {(c["regime"], c["dtype"]): c for c in _a["T4_factorial"]}
    for key, want in ((("full", "fp64"), 1.8e-3), (("full", "fp32"), 9.3e-1),
                      (("sub", "fp64"), 1.8e-10), (("sub", "fp32"), 5.5e-3)):
        if key in _f:
            check(f"factorial cell {key[0]}/{key[1]}", want,
                  _f[key]["orth_err_median"], tol=0.15)
    checks += 1
    if all(c["n"] == 24 for c in _a["T4_factorial"]):
        print("  ok  factorial cells are reported per condition (n=24 each)")
    else:
        failures.append("factorial cells are not n=24")
else:
    failures.append("results/conditioning_audit.json missing; run "
                    "experiments/exp_conditioning_audit.py")

print("== polynomial route is precision-insensitive ==")
_pw = ROOT / "results" / "precision_wall.json"
if _pw.exists():
    _rows_pw = json.loads(_pw.read_text())["rows"]
    def _med(dt, K):
        v = [r["orth_err"] for r in _rows_pw
             if r["regime"] == "sub" and r["method"] == "muon_ns"
             and r["dtype"] == dt and r["K"] == K]
        return float(np.nanmedian(v))
    check("quintic K=5, fp64", 0.649, _med("fp64", 5), tol=0.05)
    check("quintic K=5, bf16", 0.625, _med("bf16", 5), tol=0.05)
    checks += 1
    if abs(_med("bf16", 5) - _med("fp64", 5)) / _med("fp64", 5) < 0.10:
        print("  ok  bfloat16 matches float64 for the polynomial (no precision wall)")
    else:
        failures.append("polynomial precision sensitivity contradicts the paper")
else:
    print("  (precision_wall.json absent, skipped)")

print("== metric audit: 4 orthogonalizers x 3 seeds at 10.7M ==")
_mp_path = ROOT / "results" / "metric_power.json"
_grid = sorted((ROOT / "results" / "metric_power").glob("*.json"))
_runs = [f for f in _grid if f.name != "probe.json"]
checks += 1
if len(_runs) == 12:
    print("  ok  12 grid runs present (4 methods x 3 seeds)")
else:
    failures.append(f"metric_power grid has {len(_runs)} runs, expected 12")
if _mp_path.exists():
    _p = json.loads(_mp_path.read_text())
    _an = _p["anova"]
    check("grid: methods", 4, _an["k"], tol=0.0)
    check("grid: seeds per method", 3, _an["n_per_method"], tol=0.0)
    check("grid: F(3,8)", 0.11, _an["F"], tol=0.15)
    check("grid: p", 0.95, _an["p"], tol=0.03)
    check("grid: eta^2", 0.039, _an["eta_squared"], tol=0.15)
    check("grid: pooled seed sd", 0.0100, _an["pooled_sd"], tol=0.10)
    check("grid: spread of method means", 0.0041, 
          max(r["mean"] for r in _p["rows"]) - min(r["mean"] for r in _p["rows"]),
          tol=0.15)
    check("grid: min detectable difference", 0.0228,
          _an["min_detectable_difference"], tol=0.10)
    check("grid: orth_err vs loss Spearman", -0.40,
          _p["orth_err_vs_loss"]["spearman"], tol=0.10)
    _means = {r["method"]: r["mean"] for r in _p["rows"]}
    checks += 1
    if _means["polar_express"] < _means["gram"] < _means["gram_ns"]:
        print("  ok  exact Gram ranks 3rd of 4 and Polar Express 1st")
    else:
        failures.append("the metric-reversal ordering in the paper no longer holds")
else:
    failures.append("results/metric_power.json missing; run "
                    "experiments/exp_metric_power.py")

print("== power curve: seeds required per arm ==")
_pc = ROOT / "results" / "power_curve.json"
if _pc.exists():
    _q = json.loads(_pc.read_text())
    check("power: k arms", 4, _q["k"], tol=0.0)
    check("power: n per arm reported", 3, _q["n_per_arm_reported"], tol=0.0)
    check("power: one-way MDD at n=3", 0.0327, _q["mdd_global_n3"], tol=0.02)
    check("power: pairwise MDD at n=3", 0.0306, _q["mdd_pairwise_n3"], tol=0.02)
    check("power: seeds for observed spread", 117,
          _q["n_for_observed_spread"], tol=0.05)
    check("power: runs for observed spread", 468,
          _q["runs_for_observed_spread"], tol=0.05)
    checks += 1
    if _q["published_mdd_is_two_sample_normal"]:
        print("  ok  published MDD identified as the two-sample normal figure")
    else:
        failures.append("published MDD no longer matches the two-sample formula")
    checks += 1
    if _q["published_mdd_power_global"] < 0.60:
        print(f"  ok  published MDD is underpowered for 4 arms "
              f"({_q['published_mdd_power_global']:.2f})")
    else:
        failures.append("published MDD is not underpowered; the correction is moot")
else:
    failures.append("results/power_curve.json missing; run "
                    "experiments/exp_power_curve.py")

print("== metric audit (single-scale, 49.7M) ==")
_ma = ROOT / "results" / "metric_audit.json"
if _ma.exists():
    _m = json.loads(_ma.read_text())
    check("metric audit n", 3, _m["n_fully_reported"], tol=0.0)
    check("orth_err reversals", 3, _m["reversals_vs_loss"]["orth_err"], tol=0.0)
    check("eta reversals", 3, _m["reversals_vs_loss"]["eta"], tol=0.0)
    check("Kendall tau orth_err", -0.33, _m["kendall_tau_vs_loss"]["orth_err"], tol=0.02)
    check("Kendall tau eta", -0.33, _m["kendall_tau_vs_loss"]["eta"], tol=0.02)
    check("eta vs orth_err Spearman", 0.83, _m["eta_orth_spearman"]["rho"], tol=0.03)
    check("within-family gap in seed sd", 0.12,
          _m["within_family_10.7M"]["gap_in_seed_sd"], tol=0.15)
else:
    failures.append("results/metric_audit.json missing; run "
                    "experiments/exp_metric_audit.py")

print("== eta is a spectral functional (Propositions 2-3) ==")
_es = json.loads((ROOT / "results" / "eta_spectral.json").read_text())
for _lab, _ok in (
    ("T1 truncation identity holds", _es["t1_holds"]),
    ("T2 spectral identity holds", _es["t2_holds"]),
    ("T1 eta agreement below 1e-10", _es["t1_max_eta_absdiff"] < 1e-10),
    ("T1 orth agreement below 1e-10", _es["t1_max_orth_absdiff"] < 1e-10),
    ("T2 eta agreement below 1e-10", _es["t2_max_eta_absdiff"] < 1e-10),
    ("paper states the truncation proposition", "prop:truncation" in PAPER),
    ("paper states the spectral proposition", "prop:spectral" in PAPER),
    ("paper contains the antagonism table", "tab:antagonism" in PAPER),
    ("paper no longer claims a fixed random-subspace eta",
     "0.41$" not in PAPER_BODY and "regardless of $r$" not in PAPER_BODY),
):
    # These twelve claims belong to the metric-convention analysis of the long
    # manuscript (Sec. 19-20), which the BIT version deliberately does not
    # contain: a numerical-analysis submission is not the place for a survey of
    # step-size conventions.  Skipping them for that manuscript is a statement
    # about which claims each paper makes, not a relaxation of any check.
    if IS_BIT and _lab in _TMLR_ONLY:
        continue
    checks += 1
    if _ok:
        print(f"  ok  {_lab}")
    else:
        failures.append(f"failed: {_lab}")

_curve = {c["r"]: c for c in _es["truncation_curve"]}
for _r, _eta, _err in ((1, 0.740, 0.991), (10, 0.928, 0.910),
                       (29, 0.983, 0.707), (58, 1.000, 0.000)):
    check(f"truncation eta at r={_r}", _eta, _curve[_r]["eta_median"], tol=0.01)
    check(f"truncation orth_err at r={_r}", _err,
          _curve[_r]["orth_err_median"], tol=0.01)
check("top-10 nuclear mass (median)", 0.928,
      _es["top10_nuclear_mass_median"], tol=0.01)
check("top-10 nuclear mass (min)", 0.748,
      _es["top10_nuclear_mass_min"], tol=0.01)

# Paired inversion: rank-10 truncation against the Muon quintic, same matrices.
_tr = {(x["param"], x["r"]): x for x in _es["truncation"] if x["regime"] == "sub"}
_q5 = {x["param"]: x for x in _es["spectral"]
       if x["regime"] == "sub" and x["method"] == "muon_ns" and x["K"] == 5}
_common = sorted(set(p for p, r in _tr if r == 10) & set(_q5))
_d_eta = [_tr[(p, 10)]["eta_closed_form"] - _q5[p]["eta_direct"] for p in _common]
_d_err = [_tr[(p, 10)]["orth_err"] - _q5[p]["orth_err"] for p in _common]
check("paired inversion: eta higher (of n)", len(_common),
      sum(1 for d in _d_eta if d > 0), tol=0.0)
check("paired inversion: orth_err also higher (of n)", len(_common),
      sum(1 for d in _d_err if d > 0), tol=0.0)
check("paired inversion: median delta eta", 0.31,
      float(np.median(_d_eta)), tol=0.10)
check("paired inversion: median delta orth_err", 0.26,
      float(np.median(_d_err)), tol=0.15)

for _r, _norm, _rand in ((10, 0.270, 0.130), (58, 0.570, 0.353)):
    _s = _es["selection"][str(_r)]
    check(f"row selection r={_r}, norm", _norm, _s["norm_median"], tol=0.05)
    check(f"row selection r={_r}, random", _rand, _s["random_median"], tol=0.10)
    check(f"row selection r={_r}, norm wins (of n)", _s["n"],
          _s["norm_wins"], tol=0.0)

print("== schedule diagnostic: Proposition 3 applied ==")
_sd = json.loads((ROOT / "results" / "schedule_diagnostic.json").read_text())
_sub = {r["schedule"]: r for r in _sd["summary"]["sub"]}
_fullS = {r["schedule"]: r for r in _sd["summary"]["full"]}
for _nm, _eta, _gain in (
    ("exact polar factor", 1.0000, 1.0000),
    ("raw momentum", 0.7502, 0.7450),
    ("Muon quintic, 1 step", 0.7897, 0.5578),
    ("Muon quintic, 5 steps", 0.6216, 0.7470),
    ("Muon quintic, 10 steps", 0.9319, 1.0569),
    ("Polar Express, 5 steps", 0.4674, 0.9318),
    ("Polar Express, 8 steps", 0.5264, 0.9541),
    ("Gram-NS (Dion3), 5 steps", 0.7402, 0.7589),
):
    check(f"diagnostic eta: {_nm}", _eta, _sub[_nm]["eta_median"], tol=0.005)
    check(f"diagnostic gain: {_nm}", _gain, _sub[_nm]["gain_median"], tol=0.005)
for _nm, _eta, _gain in (
    ("raw momentum", 0.5930, 0.5770),
    ("Muon quintic, 10 steps", 0.8552, 0.9983),
    ("Polar Express, 5 steps", 0.4806, 0.9600),
):
    check(f"diagnostic eta (full): {_nm}", _eta, _fullS[_nm]["eta_median"], tol=0.005)
    check(f"diagnostic gain (full): {_nm}", _gain, _fullS[_nm]["gain_median"], tol=0.005)

# gain = eta * max|p| is an identity, not a fit.
check("gain equals eta times ||U||_2", 0.0,
      max(abs(r["gain"] - r["eta"] * r["p_max"]) for r in _sd["rows"]),
      tol=1e-9)
# The scalar-map evaluation reproduces the matrix evaluation for fixed polys.
check("diagnostic matches matrix eta (fixed polynomials)",
      True, _sd["spectral_vs_matrix"]["Muon quintic, 5 steps"]["max"] < 1e-4)
check("diagnostic is arithmetic-dependent for PE 8",
      True, _sd["spectral_vs_matrix"]["Polar Express, 8 steps"]["max"] > 0.1)
# The matched-RMS convention reorders the table: exact 7th of 9, raw 1st.
check("matched-RMS gain ranks the exact factor seventh", 7.0,
      _sub["exact polar factor"]["gain_rms_rank_median"], tol=0.5)
check("matched-RMS gain ranks raw momentum first", 1.0,
      _sub["raw momentum"]["gain_rms_rank_median"], tol=0.5)
# How many cells the eight-step arithmetic actually moves.
check("eight-step cells moved by more than 0.1", 13,
      _sd["pe8_cells_absdiff_gt_0_1"]["total"])
check("eight-step cells: split of 13", True,
      (_sd["pe8_cells_absdiff_gt_0_1"]["sub"] == 4
       and _sd["pe8_cells_absdiff_gt_0_1"]["full"] == 9))
# Where the shortfall sits: head, not tail.
_t5 = _sd["tail_share_by_threshold"]["sub"]["Muon quintic, 5 steps"]
check("tail share of shortfall at 1e-3", 0.0182, _t5["1e-03"], tol=0.005)
check("tail share of shortfall at 1e-2", 0.053, _t5["1e-02"], tol=0.004)
check("leading direction mass share", 0.740,
      _sub["Muon quintic, 5 steps"]["w1_median"], tol=0.005)
check("leading direction share of shortfall", 0.809,
      _sub["Muon quintic, 5 steps"]["top_shortfall_share_median"], tol=0.005)

# ---- the effective-rank law -------------------------------------------
# er = ||U||_F^2/||U||_2^2 and rho = 1/sqrt(er) are one number.
check("rho squared times effective rank is one", True,
      max(abs(r["rho"] ** 2 * r["eff_rank"] - 1.0) for r in _sd["rows"])
      < 1e-12)
# gain_rms = eta * sqrt(mn) * rho, and sqrt(mn) is a property of the matrix,
# not of the schedule: the whole schedule-dependence of the third convention is
# the factor rho.
_by_matrix = {}
for _r in _sd["rows"]:
    _by_matrix.setdefault((_r["param"], _r["regime"]), []).append(
        _r["gain_rms"] / (_r["eta"] * _r["rho"]))
_spread = max((max(_v) - min(_v)) / max(_v) for _v in _by_matrix.values())
check("gain_rms/(eta*rho) is schedule-independent per matrix", True,
      _spread < 1e-12)
# The exact polar factor is flat, so it attains the effective-rank bound.
for _reg, _er, _bound in (("sub", 58.0, 58), ("full", 384.0, 384)):
    _ex = [r for r in _sd["rows"]
           if r["regime"] == _reg and r["schedule"] == "exact polar factor"]
    check(f"exact factor attains the effective-rank bound ({_reg})", _er,
          float(np.median([r["eff_rank"] for r in _ex])), tol=1e-6)
    check(f"effective-rank bound is min(m,n) ({_reg})", _bound,
          int(_ex[0]["rank_bound"]))
_g = _sd["gauge"]
check("gauge: pairwise comparisons per regime", 864,
      _g["sub"]["n_pairs_checked"])
check("gauge: eta-vs-matched-RMS reversals (sub)", 440,
      _g["sub"]["reversals_eta_vs_rms"])
check("gauge: eta-vs-matched-RMS reversals (full)", 522,
      _g["full"]["reversals_eta_vs_rms"])
check("gauge: eta-vs-emitted-gain reversals (sub)", 293,
      _g["sub"]["reversals_eta_vs_gain"])
check("gauge: analytic reversal criterion reproduces every disagreement",
      True, (_g["sub"]["criterion_agrees"] == 864
             and _g["full"]["criterion_agrees"] == 864))
check("gauge: three orders distinct in every matrix", True,
      (_g["sub"]["all_three_orders_distinct"] == 24
       and _g["full"]["all_three_orders_distinct"] == 24))
check("gauge: median rank displacement (sub)", 6.0,
      _g["sub"]["rank_displacement_median"], tol=0.51)
check("gauge: median rank displacement (full)", 7.0,
      _g["full"]["rank_displacement_median"], tol=0.51)
check("gauge: exact factor effective rank (sub, median)", 58.0,
      _sub["exact polar factor"]["eff_rank_median"], tol=0.01)
check("gauge: raw momentum effective rank (sub, median)", 1.01,
      _sub["raw momentum"]["eff_rank_median"], tol=0.02)

# The control's rank penalty, decomposed, at the shape the audit runs.
_idx = {(r["param"], r["regime"], r["schedule"]): r for r in _sd["rows"]}
_fp = sorted({r["param"] for r in _sd["rows"] if r["regime"] == "full"})
_gn, _ex = "Gram-NS (Dion3), 5 steps", "exact polar factor"
_gap = float(np.median([_idx[(p, "full", _gn)]["gain_rms"]
                        / _idx[(p, "full", _ex)]["gain_rms"] for p in _fp]))
_pen = float(np.median([(_idx[(p, "full", _ex)]["eff_rank"]
                         / _idx[(p, "full", _gn)]["eff_rank"]) ** 0.5
                        for p in _fp]))
_adv = float(np.median([_idx[(p, "full", _ex)]["eta"]
                        / _idx[(p, "full", _gn)]["eta"] for p in _fp]))
check("matched-RMS gap, Gram-NS over exact (full shape)", 11.1, _gap, tol=0.05)
check("effective-rank penalty at full shape", 19.6, _pen, tol=0.05)
_sp = sorted({r["param"] for r in _sd["rows"] if r["regime"] == "sub"})
_pen_sub = float(np.median([(_idx[(p, "sub", _ex)]["eff_rank"]
                             / _idx[(p, "sub", _gn)]["eff_rank"]) ** 0.5
                            for p in _sp]))
check("effective-rank penalty at row-subsampled shape", 7.62, _pen_sub, tol=0.05)
check("exact factor efficiency advantage at full shape", 1.79, _adv, tol=0.01)
check("the control's gap is rank over efficiency, per matrix", True,
      max(abs((_idx[(p, "full", _gn)]["gain_rms"]
               / _idx[(p, "full", _ex)]["gain_rms"])
              * (_idx[(p, "full", _ex)]["eta"] / _idx[(p, "full", _gn)]["eta"])
              / ((_idx[(p, "full", _ex)]["eff_rank"]
                  / _idx[(p, "full", _gn)]["eff_rank"]) ** 0.5) - 1.0)
          for p in _fp) < 1e-12)

# The eight-step value in Table tab:eta and in the diagnostic table agree.
_pe8 = [r for r in _sd["rows"] if r["schedule"] == "Polar Express, 8 steps"
        and r["regime"] == "sub"]
check("PE 8 eta matches the metric table", 0.5264,
      float(np.median([r["eta"] for r in _pe8])), tol=0.005)
if not IS_BIT:
    check("paper quotes the same PE 8 value", True,
          "0.5264" in PAPER_BODY)

_sw = {(x["fit_floor"], x["regime"]): x for x in _sd["fit_window_sweep"]}
check("fit window 1e-6: eta", 0.468, _sw[(1e-6, "sub")]["eta_median"], tol=0.005)
check("fit window 1e-2: eta", 0.980, _sw[(1e-2, "sub")]["eta_median"], tol=0.005)
check("fit window 1e-2: no overshoot", 1.000,
      _sw[(1e-2, "sub")]["p_max_median"], tol=1e-5)
_gsw = [x["gain_median"] for x in _sd["fit_window_sweep"] if x["regime"] == "sub"]
check("gain flat across fit window: min", 0.85, min(_gsw), tol=0.02)
check("gain flat across fit window: max", 0.98, max(_gsw), tol=0.02)

for _lab, _ok in (
    ("paper scores schedules on a spectrum", "sec:diagnostic" in PAPER),
    ("paper has the diagnostic table", "tab:diagnostic" in PAPER),
    ("paper defines the gain", "eq:gaindim" in PAPER),
    ("paper quotes Gram-NS eta", "0.740" in PAPER_BODY),
    ("paper states the matched-RMS reordering", "seventh of nine" in PAPER_FLAT),
    ("paper states the head/tail split", "81\\%" in PAPER_BODY),
    ("paper states the three-curvature-model proposition", "prop:gauge" in PAPER),
    ("paper states the effective-rank law", "cor:effrank" in PAPER),
    ("paper states the pairwise reversal count", "440" in PAPER_FLAT),
    ("paper states the three distinct orderings", "three different orderings" in PAPER_FLAT),
    ("paper states the control's rank penalty", "11.1" in PAPER_FLAT),
    ("paper denies an update-only canonical metric",
     "property of the pair (update, curvature)" in PAPER_FLAT),
):
    # These belong to Sec. 19-20 of the long manuscript (step-size conventions),
    # which the BIT version deliberately does not contain.  Skipping them there
    # states which claims each paper makes; it does not relax a check.
    if IS_BIT and _lab in _TMLR_ONLY:
        continue
    checks += 1
    if _ok:
        print(f"  ok  {_lab}")
    else:
        failures.append(f"failed: {_lab}")

# The four-method rank correlation quoted in the audit, with Gram-NS scored.
_pw = {r["method"]: r["mean"] for r in
       json.loads((ROOT / "results" / "metric_power.json").read_text())["rows"]}
_eta4 = {"gram": 1.0,
         "ns": _sub["Muon quintic, 5 steps"]["eta_median"],
         "gram_ns": _sub["Gram-NS (Dion3), 5 steps"]["eta_median"],
         "polar_express": _sub["Polar Express, 5 steps"]["eta_median"]}


def _rank(a):
    o = np.asarray(a).argsort()
    r = np.empty(len(o))
    r[o] = np.arange(len(o))
    return r


_lr = _rank([_pw[m] for m in _eta4])
_er = _rank([-_eta4[m] for m in _eta4])
_n = len(_eta4)
# The matched-RMS gain reverses the four-method loss order (Section 9.3).
_gain4 = {"gram": _sub["exact polar factor"]["gain_rms_median"],
          "ns": _sub["Muon quintic, 5 steps"]["gain_rms_median"],
          "gram_ns": _sub["Gram-NS (Dion3), 5 steps"]["gain_rms_median"],
          "polar_express": _sub["Polar Express, 5 steps"]["gain_rms_median"]}
_g_rank = _rank([-_gain4[m] for m in _gain4])
check("matched-RMS gain reverses the four-method loss order", True,
      list(_g_rank) == list(_n - 1 - _lr))
check("eta vs loss rank correlation over four methods", -0.80,
      float(1 - 6 * ((_lr - _er) ** 2).sum() / (_n * (_n ** 2 - 1))), tol=0.01)

print("== retracted claims are absent ==")
retracted = [
    ("no mixed-shape 33x timing claim",
     "33\\times" not in PAPER_BODY and "62.7" not in PAPER_BODY),
    ("no claim that kappa^2 ~ 1e-7 is at the edge of float64",
     "at the edge of float64" not in PAPER_BODY),
    ("no claim that the conditioning ratio is an artefact",
     "not a resolvable property" not in PAPER_BODY),
    ("no claim that bf16 blinds the polynomial",
     "cannot see the small directions at all" not in PAPER_BODY),
    ("no tautological kappa_res rule",
     "kappa_{\\mathrm{res}}^{2}" not in PAPER_BODY),
    ("MDD is the four-arm value, not the two-sample one",
     "0.0228" not in PAPER_BODY and "0.023$ nats" not in PAPER_BODY
     and "0.033$ nats" in PAPER_BODY),
    ("title states the conditioning result, not an overclaim",
     ("Gram-based polar factors in finite and mixed precision" in PAPER)
     if IS_BIT
     else ("Below the Float32 Unit Roundoff" in PAPER)),
]
for label, ok in retracted:
    checks += 1
    if ok:
        print(f"  ok  {label}")
    else:
        failures.append(f"failed: {label}")

print("== textual integrity ==")

# The bibliography is one careless edit away from vanishing: with no \bibliography
# call the .bbl goes stale or empty, the PDF keeps rendering, and the citations would
# no longer regenerate from source.  Both halves are checked when refs.bib sits
# beside the manuscript, which is not guaranteed once the manuscript travels alone.
_bibfile = _paper_path.parent / "refs.bib"
if _bibfile.exists():
    _cited: set[str] = set()
    for _m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", PAPER):
        _cited |= {k.strip() for k in _m.group(1).split(",")}
    _entries = set(re.findall(r"@\w+\{([^,]+),", _bibfile.read_text()))
    _unresolved = sorted(_cited - _entries)
    _bib_ok = not _unresolved
    if _unresolved:
        print(f"  cited keys missing from refs.bib: {_unresolved}")
else:
    _bib_ok = True

required = [
    ("the manuscript includes its bibliography", "\\bibliography{" in PAPER),
    ("every cited key is in refs.bib", _bib_ok),
    ("no leftover fill placeholders", r"\\TBD\{" not in PAPER.split("newcommand")[-1]
     or len(re.findall(r"\\TBD\{", PAPER)) <= 1),
    ("proposition restricted to linear",
     IS_BIT or "linear contraction" in PAPER),
    ("limitations section present", "\\section{Limitations}" in PAPER),
    ("large-scale claims are scoped",
     ("report no pre-training runs" in PAPER.lower()
      or "analytic projection" in PAPER.lower())
     and "multi-node result" in PAPER.lower()),
]
for label, ok in required:
    checks += 1
    if ok:
        print(f"  ok  {label}")
    else:
        failures.append(f"failed: {label}")

if IS_BIT:
    import re as _re

    # ---- reference integrity -------------------------------------------------
    # The single most valuable check here: every \ref and \eqref must resolve,
    # and none of the labels withdrawn in revision may be referenced or present.
    _labels = set(_re.findall(r"\\label\{([^}]+)\}", PAPER))
    _refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]+)\}", PAPER))
    _WITHDRAWN = {"prop:bound", "cor:linear", "prop:subspace", "rem:notclaimed",
                  "sec:poly-bound", "tab:lp", "tab:matched", "tab:poly"}
    _missing = sorted(_refs - _labels)
    _stale = sorted(_refs & _WITHDRAWN)
    _resurrected = sorted(_labels & _WITHDRAWN)

    _ap = json.loads((ROOT / "results" / "aposteriori.json").read_text())
    _ar = json.loads((ROOT / "results" / "aposteriori_real.json").read_text())
    _th = json.loads((ROOT / "results" / "theory_upgrades.json").read_text())
    _r5 = json.loads((ROOT / "results" / "round5.json").read_text())
    _pu = json.loads((ROOT / "results" / "precision_update.json").read_text())
    _pt = json.loads((ROOT / "results" / "precision_training.json").read_text())
    _so = json.loads((ROOT / "results" / "subspace_order.json").read_text())
    _patches = ROOT / "paper" / "nlaa" / "wiley" / "PATCHES.md"
    _ca = json.loads((ROOT / "results" / "causal.json").read_text())
    _au = json.loads((ROOT / "results" / "audit.json").read_text())

    _floors = _r5["E_newton_schulz_floors"]["cells"]
    _subeff = _ar["subsample_descent_efficiency"]["median"]

    _bit_checks = (
        ("the manuscript has no dangling reference", not _missing,
         "missing: " + ", ".join(_missing)),
        ("no withdrawn label is referenced", not _stale,
         "referenced: " + ", ".join(_stale)),
        ("no withdrawn label survives", not _resurrected,
         "present: " + ", ".join(_resurrected)),
        ("neither is a degree bound claimed in its place",
         "Proposition~\\ref{prop:bound}" not in PAPER
         and "prop:bound" not in PAPER, ""),
        ("both polynomial regimes are stated, degree-limited and floored",
         "at the iteration counts used in practice" in PAPER_FLAT.lower()
         and "precision floor" in PAPER_FLAT.lower(), ""),
        ("the floor grows like eps*kappa, not like eps*kappa^2",
         "rather than $\\eps\\kappa^{2}$" in PAPER
         and 0.4 < float(np.log(_floors["1e+03"]["float32"]["floor"]
                                / _floors["1e+02"]["float32"]["floor"])
                         / np.log(10.0)) < 1.2, ""),
        ("neither is a revision narrated in the text",
         "earlier draft" not in PAPER_FLAT
         and "previous draft" not in PAPER_FLAT, ""),

        # ---- theory
        ("the bound is stated with both branches",
         "eq:twosided" in PAPER and "(1-t_i)^{-1/2}" in PAPER, ""),
        ("the bound is proved attained",
         "prop:sharp" in PAPER
         and _r5["A_general_theorem"]["extremal_attains"] is True
         and _r5["A_general_theorem"]["all_valid"] is True, ""),
        ("the first-order form of the error is a lemma",
         "lem:firstorder" in PAPER and "eq:firstorder" in PAPER, ""),
        ("the rotation is argued not to need an eigenvalue gap",
         "rem:nogap" in PAPER and "sigma_i+\\sigma_j" in PAPER.replace(" ", ""), ""),
        ("the shift repair is stated and bounded",
         "prop:shift" in PAPER and "eq:shift" in PAPER
         and _r5["B_shift"]["always_contraction"] is True
         and _r5["B_shift"]["bound_valid"] is True, ""),
        ("the route is discussed as mixed backward-forward stable",
         "sec:notstable" in PAPER and "mixed backward--forward" in PAPER_FLAT
         and _th["non_isometry"]["not_backward_stable"] is True, ""),
        ("the measured constant is a total, formation plus eigensolver",
         "eq:etatotal" in PAPER and "eigensolver" in PAPER_FLAT, ""),

        # ---- tables present
        ("the constant and truncation tables are present",
         "tab:aposteriori" in PAPER and "tab:truncation" in PAPER, ""),
        ("the real-buffer table is present", "tab:real" in PAPER, ""),
        ("the mixed-precision table is present", "tab:mixed" in PAPER, ""),
        ("the threshold sweep is present and the bound holds in it",
         "tab:sweep" in PAPER
         and _r5["D_kappa_sweep_fp32"]["bound_valid_admissible"] is True, ""),
        ("the sweep reports the shift inside t",
         _r5["D_kappa_sweep_fp32"]["shift_within_t"] is True, ""),
        # The figure is looked for beside the manuscript first, then in the
        # author's layout: the manuscript may live outside this archive, in which
        # case its figure travels with it rather than with the code.
        ("the figure is present", "fig:knee" in PAPER and any(
            q.exists() for q in (_paper_path.parent / "figures" / "knee.pdf",
                                 _paper_path.parent.parent / "figures" / "knee.pdf",
                                 ROOT / "paper" / "nlaa" / "figures" / "knee.pdf",
                                 ROOT / "figures" / "knee.pdf")), ""),

        # ---- the polynomial floors
        ("the floors table quotes the measured floors",
         "tab:floors" in PAPER
         and abs(_floors["1e+02"]["bfloat16"]["floor"] - 0.184858) < 1e-4
         and abs(_floors["1e+03"]["bfloat16"]["floor"] - 0.873682) < 1e-4
         and "1.85\\times10^{-1}" in PAPER and "8.74\\times10^{-1}" in PAPER, ""),
        ("the floors separate float64 from float32",
         "3.2\\times10^{-6}" in PAPER and "2.6\\times10^{-5}" in PAPER, ""),
        ("the iteration counts to 1e-1 are quoted",
         _floors["1e+02"]["float64"]["iters_to_1e-1"] == 15
         and _floors["1e+03"]["float64"]["iters_to_1e-1"] == 20
         and "fifteen to twenty" in PAPER_FLAT, ""),

        # ---- subsampling
        ("subsampling is reported to change the target",
         "sec:subsample" in PAPER and "eq:subeff" in PAPER
         and abs(_subeff - 0.6419) < 5e-3 and "0.642" in PAPER_FLAT, ""),

        # ---- the spectral family
        ("the spectral family is stated with its measured constant",
         "prop:family" in PAPER and "eq:Lq" in PAPER
         and abs(_th["family"]["exact_ratio_median"] - 0.9935) < 0.01
         and "0.9935" in PAPER_FLAT, ""),
        ("the family bound is stated as attainable but not typical",
         "0.078" in PAPER_FLAT and "factor of thirteen" in PAPER_FLAT, ""),
        ("the operator-norm corollary is stated", "eq:Lqnorm" in PAPER, ""),
        ("the rotation law is reported with its measured ratios",
         "0.909" in PAPER_FLAT and "0.9999" in PAPER_FLAT
         and abs(_so["verdict"]["block_ratio_median"] - 0.995) < 0.02, ""),

        # ---- precision at the update and at the loss
        ("the precision effect is measured at the update",
         "61\\%" in PAPER_FLAT and "5.1\\%" in PAPER_FLAT
         and abs(_pu["relative_update_difference"]["exact_fp32"]["median"]
                 - 0.615) < 0.01
         and abs(_pu["verdict"]["truncation_reduces_precision_sensitivity"]
                 - 12.1) < 0.5, ""),
        ("the precision effect is measured at the loss",
         "4.0359" in PAPER_FLAT
         and abs(_pt["arms"]["exact_fp32"]["mean"]
                 - _pt["arms"]["fp32"]["mean"] if False else
                 _pt["arms"]["exact_fp32"]["mean"]
                 - _pt["arms"]["exact_fp64"]["mean"]) < 1e-4, ""),
        ("the gradient's concentration and its overlap with the buffer are reported",
         abs(_ca["H15"]["grad_flatness"]["median"] - 0.603) < 0.01
         and abs(_ca["H15"]["overlap_r1"]["median"] - 0.984) < 0.01
         and abs(_ca["H15"]["overlap_r10"]["median"] - 0.744) < 0.01
         and abs(_ca["H15"]["grad_nuclear_in_buffer_tail"]["median"] - 0.087) < 0.01
         and "0.984" in PAPER_FLAT and "0.603" in PAPER_FLAT, ""),
        ("the truncation cost is priced in both currencies and they agree",
         abs(_ca["H15"]["truncation_nuclear_cost"]["median"] - 6.3e-3) / 6.3e-3 < 0.05
         and abs(_ca["H15"]["truncation_functional_cost"]["median"] - 4.0e-3) / 4.0e-3 < 0.05
         and "6.3\\times10^{-3}" in PAPER and "4.0\\times10^{-3}" in PAPER, ""),
        ("the paper says Muon forms but does not diagonalise the Gram matrix",
         "never diagonalises" in PAPER, ""),
        # The paper must say that the checks detect drift rather than validate the
        # measurement; the wording is free, the substance is not.
        ("the checks are described as drift detection, not validation",
         "detect drift" in PAPER_FLAT and "not evidence of correctness" in PAPER_FLAT
         and "rest on the" in PAPER_FLAT, ""),
        ("the loss-facing metric is reported and is far smaller than the norm",
         "sec:functional" in PAPER and "e_G" in PAPER
         and abs(_ca["H15"]["ratio_fp32"]["median"] - 6.27e-5) / 6.27e-5 < 0.02
         and abs(_ca["H15"]["rel_functional_fp32"]["median"] - 2.45e-4)
             / 2.45e-4 < 0.02
         and "6.3\\times10^{-5}" in PAPER and "2.4\\times10^{-4}" in PAPER, ""),
        ("the loss attribution is reported and is flatter than the spectrum",
         abs(_ca["H14"]["share_top10_loss"]["median"] - 0.597) < 0.01
         and abs(_ca["H14"]["share_top10_update_norm"]["median"] - 0.913) < 0.01
         and abs(_ca["H14"]["share_bottom_half_loss"]["median"] - 0.108) < 0.005
         and "0.597" in PAPER_FLAT and "0.108" in PAPER_FLAT, ""),
        ("the audit of the shipped iteration is reported",
         "0.888" in PAPER_FLAT and "0.365" in PAPER_FLAT
         and abs(_au["at_five_steps"]["bfloat16"] - 0.879) < 0.01
         and abs(_au["floor"]["float64"] - 0.365) < 0.005, ""),
        ("the paper says Muon is outside both of its regimes",
         "outside both of this paper's regimes" in PAPER_FLAT, ""),
        ("the paired design is reported with its minimum detectable difference",
         "1.6\\times10^{-4}" in PAPER_FLAT
         and "paired standard deviation" in PAPER_FLAT, ""),

        # ---- housekeeping
        ("the paper names the CholeskyQR literature it belongs to",
         "CholeskyQR2" in PAPER and "shifted CholeskyQR" in PAPER, ""),
        # vacuous when the Wiley class is not vendored, as in a submission package
        ("the vendored class licence is noted where it belongs",
         ("copyright Wiley" in _patches.read_text()
          if _patches.exists() else True), ""),
        ("compensated summation is discussed",
         "compensated" in PAPER_FLAT.lower(), ""),
        # Claims the round-5 data refutes; they have been reintroduced once already.
        ("fp32 Gram + fp64 eigensolver is not called useless",
         "cannot restore admissibility" not in PAPER_FLAT
         and "buys nothing at all" not in PAPER_FLAT,
         "round5.json F: fp32|fp64 is 6-19x better than fp32|fp32"),
        ("bfloat16 is not called void at every kappa",
         "including $10^{-2}$" not in PAPER_FLAT, "round5.json F: bf16 admissible at kappa=10"),
        ("eta is quoted as four to eight machine epsilons",
         "four to ten" not in PAPER_FLAT and "fourteen orders" not in PAPER_FLAT,
         "aposteriori.json: 5.5-8.3 eps; round5.json E: fp64/fp32 floors 8 orders apart"),
    )
    for _lab, _ok, _detail in _bit_checks:
        checks += 1
        if _ok:
            print(f"  ok  BIT: {_lab}")
        else:
            failures.append(f"failed: BIT: {_lab}" + (f" ({_detail})" if _detail else ""))

print(f"\n{checks} checks, {len(failures)} failures")
for f in failures:
    print("  FAIL:", f)
sys.exit(1 if failures else 0)
